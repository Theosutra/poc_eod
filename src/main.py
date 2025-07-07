#!/usr/bin/env python3
"""
Pipeline RAG principal pour EODEN
Orchestration complète : Drive → Chunking → Embeddings → Vector Store
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Imports locaux
from drive_connector import DriveConnector
from embedding_manager import EmbeddingManager, ProcessedChunk
from vector_store import VectorManager, VectorDocument

# Configuration du logging
import os
os.makedirs("logs", exist_ok=True)  # Créer le dossier logs s'il n'existe pas

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/poc.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Pipeline principal RAG pour EODEN"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        # Charger la configuration
        load_dotenv()
        self.config = self._load_config(config_path)
        
        # Initialiser les composants
        self.drive_connector = None
        self.embedding_manager = None
        self.vector_manager = None
        
        self._setup_directories()
        self._init_components()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Charger la configuration YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Substituer les variables d'environnement
            config = self._substitute_env_vars(config)
            logger.info(f"Configuration chargée: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            sys.exit(1)
    
    def _substitute_env_vars(self, obj):
        """Substituer les variables d'environnement dans la config"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj
    
    def _setup_directories(self):
        """Créer les répertoires nécessaires"""
        directories = [
            "data/cache",
            "data/embeddings", 
            "data/output",
            "logs",
            "config"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _init_components(self):
        """Initialiser les composants du pipeline"""
        try:
            # Google Drive Connector
            logger.info("🔗 Initialisation Google Drive...")
            self.drive_connector = DriveConnector(
                credentials_path=self.config['google_drive']['credentials_file'],
                cache_dir=self.config['google_drive']['cache_directory']
            )
            
            # Embedding Manager
            logger.info("🧠 Initialisation Embedding Manager...")
            gemini_api_key = self.config['apis']['gemini_api_key']
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY manquant dans .env")
            
            self.embedding_manager = EmbeddingManager(
                api_key=gemini_api_key,
                cache_dir=self.config['embeddings']['cache_directory']
            )
            
            # Vector Store Manager
            logger.info("📦 Initialisation Vector Store...")
            self.vector_manager = VectorManager(config_path="config/settings.yaml")
            
            logger.info("✅ Tous les composants initialisés")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation: {e}")
            sys.exit(1)
    
    def discover_documents(self) -> List[Dict[str, Any]]:
        """Découvrir les documents dans Google Drive"""
        logger.info("🔍 Découverte des documents Google Drive...")
        
        all_files = []
        
        # Parcourir les dossiers configurés
        for folder_config in self.config['google_drive']['source_folders']:
            folder_name = folder_config['name']
            recursive = folder_config.get('recursive', True)
            
            logger.info(f"📁 Exploration du dossier: {folder_name}")
            
            # Trouver le dossier
            folder_id = self.drive_connector.get_folder_by_name(folder_name)
            if not folder_id:
                logger.warning(f"⚠️ Dossier non trouvé: {folder_name}")
                continue
            
            # Lister les fichiers
            files = self.drive_connector.list_files(folder_id, recursive=recursive)
            all_files.extend(files)
            logger.info(f"✅ {len(files)} fichiers trouvés dans {folder_name}")
        
        # Filtrer par extensions supportées
        supported_exts = self.config['google_drive']['supported_extensions']
        filtered_files = []
        
        for file_info in all_files:
            file_name = file_info['name']
            if any(file_name.lower().endswith(ext.lower()) for ext in supported_exts):
                filtered_files.append(file_info)
        
        logger.info(f"🎯 {len(filtered_files)} fichiers supportés découverts")
        return filtered_files
    
    def download_documents(self, files_info: List[Dict[str, Any]]) -> Dict[str, str]:
        """Télécharger et extraire le contenu des documents"""
        logger.info("⬇️ Téléchargement des documents...")
        
        downloaded_contents = {}
        
        with tqdm(total=len(files_info), desc="Téléchargement") as pbar:
            for file_info in files_info:
                try:
                    # Télécharger le fichier
                    cache_path = self.drive_connector.download_file(
                        file_info['id'],
                        file_info['name'],
                        file_info['mimeType']
                    )
                    
                    if cache_path:
                        # Extraire le contenu
                        content = self._extract_content(cache_path)
                        if content:
                            downloaded_contents[str(cache_path)] = content
                            logger.debug(f"✅ Contenu extrait: {file_info['name']}")
                        else:
                            logger.warning(f"⚠️ Extraction échouée: {file_info['name']}")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur téléchargement {file_info['name']}: {e}")
                
                pbar.update(1)
        
        logger.info(f"✅ {len(downloaded_contents)} documents téléchargés et extraits")
        return downloaded_contents
    
    def _extract_content(self, file_path: Path) -> str:
        """Extraire le contenu textuel d'un fichier"""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.pdf':
                return self._extract_pdf_content(file_path)
            elif file_ext == '.docx':
                return self._extract_docx_content(file_path)
            elif file_ext == '.pptx':
                return self._extract_pptx_content(file_path)
            elif file_ext == '.xlsx':
                return self._extract_xlsx_content(file_path)
            else:
                logger.warning(f"Type de fichier non supporté: {file_ext}")
                return ""
                
        except Exception as e:
            logger.error(f"Erreur extraction {file_path}: {e}")
            return ""
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extraire le contenu d'un PDF"""
        import pymupdf  # fitz
        
        text = ""
        doc = pymupdf.open(str(file_path))
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()
        
        doc.close()
        return text.strip()
    
    def _extract_docx_content(self, file_path: Path) -> str:
        """Extraire le contenu d'un Word"""
        from docx import Document
        
        doc = Document(str(file_path))
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Extraire les tableaux
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text for cell in row.cells)
                text += row_text + "\n"
        
        return text.strip()
    
    def _extract_pptx_content(self, file_path: Path) -> str:
        """Extraire le contenu d'un PowerPoint"""
        from pptx import Presentation
        
        prs = Presentation(str(file_path))
        text = ""
        
        for slide_num, slide in enumerate(prs.slides, 1):
            text += f"\n--- Slide {slide_num} ---\n"
            
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        
        return text.strip()
    
    def _extract_xlsx_content(self, file_path: Path) -> str:
        """Extraire le contenu d'un Excel"""
        import pandas as pd
        
        text = ""
        
        try:
            # Lire toutes les feuilles
            excel_file = pd.ExcelFile(str(file_path))
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(str(file_path), sheet_name=sheet_name)
                
                text += f"\n--- Feuille: {sheet_name} ---\n"
                
                # Convertir en texte structuré
                for index, row in df.iterrows():
                    row_text = " | ".join(str(value) for value in row.values if pd.notna(value))
                    if row_text.strip():
                        text += row_text + "\n"
        
        except Exception as e:
            logger.warning(f"Erreur lecture Excel {file_path}: {e}")
        
        return text.strip()
    
    def process_embeddings(self, file_contents: Dict[str, str]) -> List[ProcessedChunk]:
        """Traiter les documents : chunking + embeddings"""
        logger.info("🧠 Traitement des embeddings...")
        
        # Traitement en batch
        processed_chunks = self.embedding_manager.batch_process_documents(file_contents)
        
        logger.info(f"✅ {len(processed_chunks)} chunks traités")
        return processed_chunks
    
    def store_in_vector_db(self, processed_chunks: List[ProcessedChunk]) -> bool:
        """Stocker les chunks dans la base vectorielle"""
        logger.info("📦 Stockage dans la base vectorielle...")
        
        # Convertir en VectorDocument
        vector_docs = []
        for chunk in processed_chunks:
            vector_doc = VectorDocument(
                id=chunk.id,
                content=chunk.content,
                embedding=chunk.embedding,
                metadata={
                    "source_file": chunk.metadata.source_file,
                    "document_type": chunk.metadata.document_type.value,
                    "section_title": chunk.metadata.section_title,
                    "business_context": chunk.metadata.business_context,
                    "themes": chunk.metadata.themes,
                    "confidence_score": chunk.metadata.confidence_score,
                    "chunk_index": chunk.metadata.chunk_index,
                    "token_count": chunk.token_count,
                    "file_hash": chunk.metadata.file_hash,
                    "created_at": chunk.metadata.created_at
                }
            )
            vector_docs.append(vector_doc)
        
        # Stocker en batch
        batch_size = self.config['embeddings']['batch_size']
        total_stored = 0
        
        with tqdm(total=len(vector_docs), desc="Stockage vectoriel") as pbar:
            for i in range(0, len(vector_docs), batch_size):
                batch = vector_docs[i:i + batch_size]
                
                if self.vector_manager.add_documents(batch):
                    total_stored += len(batch)
                    logger.debug(f"Batch {i//batch_size + 1} stocké: {len(batch)} documents")
                else:
                    logger.error(f"Erreur stockage batch {i//batch_size + 1}")
                
                pbar.update(len(batch))
        
        logger.info(f"✅ {total_stored} documents stockés dans la base vectorielle")
        return total_stored == len(vector_docs)
    
    def test_search(self, query: str = "analyse financière") -> List[VectorDocument]:
        """Tester la recherche vectorielle"""
        logger.info(f"🔍 Test de recherche: '{query}'")
        
        # Générer l'embedding de la requête
        query_embedding = self.embedding_manager.get_embedding(query)
        
        # Rechercher
        results = self.vector_manager.search_similar(
            query_embedding=query_embedding,
            top_k=5
        )
        
        logger.info(f"✅ {len(results)} résultats trouvés")
        
        for i, result in enumerate(results, 1):
            logger.info(f"Résultat {i}:")
            logger.info(f"  Score: {result.score:.3f}")
            logger.info(f"  Source: {result.metadata.get('source_file', 'Unknown')}")
            logger.info(f"  Section: {result.metadata.get('section_title', 'Unknown')}")
            logger.info(f"  Contenu: {result.content[:150]}...")
            logger.info("-" * 50)
        
        return results
    
    def run_full_pipeline(self) -> bool:
        """Exécuter le pipeline complet"""
        logger.info("🚀 Démarrage du pipeline RAG complet")
        
        try:
            # 1. Découvrir les documents
            files_info = self.discover_documents()
            if not files_info:
                logger.error("❌ Aucun document trouvé")
                return False
            
            # 2. Télécharger et extraire le contenu
            file_contents = self.download_documents(files_info)
            if not file_contents:
                logger.error("❌ Aucun contenu extrait")
                return False
            
            # 3. Traitement des embeddings
            processed_chunks = self.process_embeddings(file_contents)
            if not processed_chunks:
                logger.error("❌ Aucun chunk traité")
                return False
            
            # 4. Stockage vectoriel
            success = self.store_in_vector_db(processed_chunks)
            if not success:
                logger.error("❌ Erreur stockage vectoriel")
                return False
            
            # 5. Test de recherche
            self.test_search("stratégie entreprise")
            self.test_search("chiffres financiers")
            
            logger.info("🎉 Pipeline RAG terminé avec succès!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur pipeline: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques du pipeline"""
        return {
            "cache_files": len(list(Path(self.config['google_drive']['cache_directory']).glob("*"))),
            "embedding_cache_size": len(self.embedding_manager.embedding_cache),
            "vector_store_type": self.config['vector_store']['type'],
            "index_name": self.config['vector_store']['index_name']
        }

def main():
    """Point d'entrée principal"""
    print("🚀 Pipeline RAG EODEN - Démarrage")
    
    # Vérifier les prérequis
    required_files = [
        "config/settings.yaml",
        "config/credentials.json",
        ".env"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Fichier manquant: {file_path}")
            print("Assurez-vous d'avoir configuré tous les prérequis")
            return
    
    # Créer et exécuter le pipeline
    try:
        pipeline = RAGPipeline()
        
        # Afficher les statistiques initiales
        stats = pipeline.get_stats()
        print(f"📊 Statistiques initiales: {stats}")
        
        # Exécuter le pipeline
        success = pipeline.run_full_pipeline()
        
        if success:
            # Afficher les statistiques finales
            final_stats = pipeline.get_stats()
            print(f"📊 Statistiques finales: {final_stats}")
            print("✅ Pipeline terminé avec succès!")
        else:
            print("❌ Pipeline échoué")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrompu par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()