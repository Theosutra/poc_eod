"""
Connecteur Google Drive pour récupérer les documents sources
"""

import os
import io
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriveConnector:
    """Connecteur pour Google Drive API"""
    
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    
    # Types de fichiers supportés
    SUPPORTED_MIMETYPES = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.google-apps.document': '.docx',  # Google Docs
        'application/vnd.google-apps.presentation': '.pptx',  # Google Slides
        'application/vnd.google-apps.spreadsheet': '.xlsx',  # Google Sheets
    }
    
    def __init__(self, credentials_path: str = "config/credentials.json", cache_dir: str = "data/cache"):
        self.credentials_path = credentials_path
        self.token_path = "config/token.pickle"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.service = self._authenticate()
    
    def _authenticate(self) -> Optional[Any]:
        """Authentification Google Drive avec Service Account ou OAuth"""
        
        # Vérifier si on utilise un Service Account
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                from google.oauth2 import service_account
                
                credentials = service_account.Credentials.from_service_account_file(
                    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                    scopes=self.SCOPES
                )
                
                service = build('drive', 'v3', credentials=credentials)
                
                # Test de connexion
                service.about().get(fields="user").execute()
                logger.info("✅ Connexion Google Drive réussie (Service Account)")
                return service
                
            except Exception as e:
                logger.error(f"❌ Erreur authentification Service Account: {e}")
                raise
        
        # Sinon, utiliser OAuth classique
        creds = None
        
        # Charger le token existant
        if os.path.exists(self.token_path):
            try:
                with open(self.token_path, 'rb') as token:
                    creds = pickle.load(token)
                logger.info("Token OAuth existant chargé")
            except Exception as e:
                logger.warning(f"Erreur chargement token OAuth: {e}")
        
        # Vérifier/renouveler les credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    logger.info("Token OAuth renouvelé")
                except Exception as e:
                    logger.error(f"Erreur renouvellement token OAuth: {e}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Fichier credentials manquant: {self.credentials_path}\n"
                        "Pour Service Account:\n"
                        "1. Aller sur Google Cloud Console\n"
                        "2. Activer Drive API\n"
                        "3. Créer un Service Account\n"
                        "4. Télécharger le JSON et le placer dans config/credentials.json\n"
                        "5. Définir GOOGLE_APPLICATION_CREDENTIALS=config/credentials.json dans .env\n\n"
                        "Pour OAuth:\n"
                        "1. Créer des credentials OAuth2 Desktop\n"
                        "2. Télécharger le JSON et le placer dans config/credentials.json"
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES
                )
                creds = flow.run_local_server(port=0)
                logger.info("Nouvelle authentification OAuth effectuée")
            
            # Sauvegarder le token OAuth
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        try:
            service = build('drive', 'v3', credentials=creds)
            # Test de connexion
            service.about().get(fields="user").execute()
            logger.info("✅ Connexion Google Drive réussie (OAuth)")
            return service
        except Exception as e:
            logger.error(f"❌ Erreur connexion Drive: {e}")
            raise
    
    def list_folders(self, parent_folder_id: str = None) -> List[Dict[str, Any]]:
        """Liste les dossiers dans un dossier parent"""
        try:
            query = "mimeType='application/vnd.google-apps.folder'"
            if parent_folder_id:
                query += f" and parents in '{parent_folder_id}'"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, parents, modifiedTime)"
            ).execute()
            
            folders = results.get('files', [])
            logger.info(f"Trouvé {len(folders)} dossiers")
            return folders
            
        except HttpError as e:
            logger.error(f"Erreur liste dossiers: {e}")
            return []
    
    def list_files(self, folder_id: str = None, recursive: bool = True) -> List[Dict[str, Any]]:
        """Liste tous les fichiers supportés dans un dossier"""
        files = []
        
        try:
            # Construire la requête
            mime_types = list(self.SUPPORTED_MIMETYPES.keys())
            mime_query = " or ".join([f"mimeType='{mt}'" for mt in mime_types])
            
            query = f"({mime_query})"
            if folder_id:
                query += f" and parents in '{folder_id}'"
            
            # Exécuter la requête
            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, parents, modifiedTime, size)",
                pageSize=1000
            ).execute()
            
            files = results.get('files', [])
            
            # Si récursif, explorer les sous-dossiers
            if recursive:
                folders = self.list_folders(folder_id)
                for folder in folders:
                    subfolder_files = self.list_files(folder['id'], recursive=True)
                    files.extend(subfolder_files)
            
            logger.info(f"Trouvé {len(files)} fichiers supportés")
            return files
            
        except HttpError as e:
            logger.error(f"Erreur liste fichiers: {e}")
            return []
    
    def download_file(self, file_id: str, file_name: str, mime_type: str) -> Optional[Path]:
        """Télécharge un fichier et le met en cache"""
        try:
            # Définir l'extension
            extension = self.SUPPORTED_MIMETYPES.get(mime_type, '.bin')
            cache_filename = f"{file_id}_{file_name}{extension}"
            cache_path = self.cache_dir / cache_filename
            
            # Vérifier si déjà en cache
            if cache_path.exists():
                logger.info(f"Fichier déjà en cache: {cache_filename}")
                return cache_path
            
            # Télécharger selon le type
            if mime_type.startswith('application/vnd.google-apps'):
                # Fichiers Google (Docs, Sheets, Slides)
                export_mime_type = self._get_export_mime_type(mime_type)
                request = self.service.files().export_media(
                    fileId=file_id, mimeType=export_mime_type
                )
            else:
                # Fichiers normaux
                request = self.service.files().get_media(fileId=file_id)
            
            # Télécharger
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    logger.info(f"Téléchargement {file_name}: {int(status.progress() * 100)}%")
            
            # Sauvegarder en cache
            with open(cache_path, 'wb') as f:
                f.write(fh.getvalue())
            
            logger.info(f"Fichier téléchargé: {cache_filename}")
            return cache_path
            
        except HttpError as e:
            logger.error(f"Erreur téléchargement {file_name}: {e}")
            return None
    
    def _get_export_mime_type(self, google_mime_type: str) -> str:
        """Convertit les types MIME Google en types exportables"""
        export_map = {
            'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        }
        return export_map.get(google_mime_type, google_mime_type)
    
    def get_folder_by_name(self, folder_name: str, parent_id: str = None) -> Optional[str]:
        """Trouve un dossier par nom"""
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if parent_id:
                query += f" and parents in '{parent_id}'"
            
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            folders = results.get('files', [])
            
            if folders:
                logger.info(f"Dossier trouvé: {folder_name}")
                return folders[0]['id']
            else:
                logger.warning(f"Dossier non trouvé: {folder_name}")
                return None
                
        except HttpError as e:
            logger.error(f"Erreur recherche dossier {folder_name}: {e}")
            return None
    
    def batch_download(self, folder_name: str = None, folder_id: str = None) -> List[Path]:
        """Télécharge tous les fichiers d'un dossier"""
        if folder_name and not folder_id:
            folder_id = self.get_folder_by_name(folder_name)
            if not folder_id:
                logger.error(f"Dossier non trouvé: {folder_name}")
                return []
        
        files = self.list_files(folder_id)
        downloaded_files = []
        
        logger.info(f"Début téléchargement de {len(files)} fichiers...")
        
        for file_info in files:
            cache_path = self.download_file(
                file_info['id'], 
                file_info['name'], 
                file_info['mimeType']
            )
            if cache_path:
                downloaded_files.append(cache_path)
        
        logger.info(f"Téléchargement terminé: {len(downloaded_files)} fichiers")
        return downloaded_files
    
    def clear_cache(self):
        """Vide le cache de téléchargement"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache vidé")

# Fonction utilitaire pour tester la connexion
def test_connection():
    """Test rapide de la connexion Drive"""
    try:
        drive = DriveConnector()
        
        # Lister quelques dossiers racine
        folders = drive.list_folders()
        print(f"✅ Connexion réussie ! Trouvé {len(folders)} dossiers racine:")
        for folder in folders[:5]:  # Afficher les 5 premiers
            print(f"  📁 {folder['name']} (ID: {folder['id']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur connexion: {e}")
        return False

if __name__ == "__main__":
    test_connection()