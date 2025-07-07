#!/usr/bin/env python3
"""
Script de démarrage rapide du POC EODEN
Vérifie la configuration et lance le pipeline RAG
"""

import os
import sys
from pathlib import Path
import yaml
from dotenv import load_dotenv

def check_requirements():
    """Vérifier les prérequis"""
    print("🔍 Vérification des prérequis...")
    
    issues = []
    
    # Vérifier les fichiers de configuration
    required_files = {
        ".env": "Fichier des variables d'environnement",
        "config/settings.yaml": "Configuration principale", 
        "config/credentials.json": "Credentials Google Drive"
    }
    
    for file_path, description in required_files.items():
        if not Path(file_path).exists():
            issues.append(f"❌ {description} manquant: {file_path}")
        else:
            print(f"✅ {description}: {file_path}")
    
    # Vérifier les variables d'environnement
    load_dotenv()
    required_env = {
        "GEMINI_API_KEY": "Clé API Gemini (Google AI Studio)",
        "PINECONE_API_KEY": "Clé API Pinecone (optionnel si Elasticsearch)",
    }
    
    for env_var, description in required_env.items():
        if not os.getenv(env_var):
            if env_var == "PINECONE_API_KEY":
                print(f"⚠️ {description} manquant (OK si utilisation Elasticsearch)")
            else:
                issues.append(f"❌ {description} manquant: {env_var}")
        else:
            print(f"✅ {description}: défini")
    
    # Vérifier les modules Python
    required_modules = [
        ("google.generativeai", "google-generativeai"),
        ("google.auth", "google-auth"), 
        ("googleapiclient", "google-api-python-client"),
        ("pymupdf", "pymupdf"),
        ("docx", "python-docx"),  # Module docx, package python-docx
        ("pptx", "python-pptx"),  # Module pptx, package python-pptx
        ("openpyxl", "openpyxl"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("tqdm", "tqdm")
    ]
    
    missing_modules = []
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ Module: {package_name}")
        except ImportError:
            missing_modules.append(package_name)
            issues.append(f"❌ Module manquant: {package_name}")
    
    if issues:
        print(f"\n🚨 {len(issues)} problèmes détectés:")
        for issue in issues:
            print(f"  {issue}")
        
        print(f"\n📋 Actions à effectuer:")
        
        if missing_modules:
            print(f"1. Installer les modules manquants:")
            print(f"   pip install {' '.join(missing_modules)}")
        
        if not Path(".env").exists():
            print(f"2. Créer le fichier .env avec vos clés API:")
            print(f"   GEMINI_API_KEY=your_gemini_key")
            print(f"   PINECONE_API_KEY=your_pinecone_key")
        
        if not Path("config/credentials.json").exists():
            print(f"3. Télécharger credentials.json depuis Google Cloud Console")
            print(f"   et le placer dans config/credentials.json")
        
        return False
    
    print(f"\n✅ Tous les prérequis sont satisfaits!")
    return True

def setup_vector_store():
    """Configurer la base vectorielle"""
    print(f"\n📦 Configuration de la base vectorielle...")
    
    try:
        with open("config/settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        store_type = config['vector_store']['type']
        print(f"Type configuré: {store_type}")
        
        if store_type == "pinecone":
            # Vérifier Pinecone
            try:
                import pinecone
                api_key = os.getenv("PINECONE_API_KEY")
                if api_key:
                    print("✅ Pinecone configuré")
                    return True
                else:
                    print("❌ PINECONE_API_KEY manquant")
                    return False
            except ImportError:
                print("❌ Module pinecone-client manquant")
                print("   pip install pinecone-client")
                return False
        
        elif store_type == "elasticsearch":
            # Vérifier Elasticsearch
            try:
                from elasticsearch import Elasticsearch
                hosts = config['vector_store']['elasticsearch']['hosts']
                es = Elasticsearch(hosts)
                if es.ping():
                    print("✅ Elasticsearch accessible")
                    return True
                else:
                    print(f"❌ Elasticsearch non accessible sur {hosts}")
                    return False
            except ImportError:
                print("❌ Module elasticsearch manquant")
                print("   pip install elasticsearch")
                return False
            except Exception as e:
                print(f"❌ Erreur connexion Elasticsearch: {e}")
                return False
        
        else:
            print(f"❌ Type de base vectorielle non supporté: {store_type}")
            return False
    
    except Exception as e:
        print(f"❌ Erreur configuration: {e}")
        return False

def test_google_drive():
    """Tester la connexion Google Drive"""
    print(f"\n🔗 Test de connexion Google Drive...")
    
    try:
        sys.path.append('src')
        from drive_connector import test_connection
        
        if test_connection():
            print("✅ Connexion Google Drive réussie")
            return True
        else:
            print("❌ Connexion Google Drive échouée")
            return False
    
    except Exception as e:
        print(f"❌ Erreur test Google Drive: {e}")
        return False

def run_pipeline():
    """Exécuter le pipeline principal"""
    print(f"\n🚀 Lancement du pipeline RAG...")
    
    try:
        sys.path.append('src')
        from main import RAGPipeline
        
        # Créer et exécuter le pipeline
        pipeline = RAGPipeline()
        success = pipeline.run_full_pipeline()
        
        if success:
            print(f"\n🎉 Pipeline terminé avec succès!")
            
            # Afficher quelques statistiques
            stats = pipeline.get_stats()
            print(f"\n📊 Statistiques:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
            return True
        else:
            print(f"\n❌ Pipeline échoué")
            return False
    
    except Exception as e:
        print(f"❌ Erreur pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Point d'entrée principal"""
    print("=" * 60)
    print("🚀 POC EODEN - Génération automatique de notes d'investissement")
    print("📋 Pipeline RAG: Drive → Chunking → Embeddings → Vector Store")
    print("=" * 60)
    
    # 1. Vérifier les prérequis
    if not check_requirements():
        print(f"\n❌ Prérequis non satisfaits. Corrigez les problèmes et relancez.")
        return 1
    
    # 2. Configurer la base vectorielle
    if not setup_vector_store():
        print(f"\n❌ Configuration base vectorielle échouée.")
        return 1
    
    # 3. Tester Google Drive
    if not test_google_drive():
        print(f"\n❌ Test Google Drive échoué.")
        return 1
    
    # 4. Demander confirmation
    print(f"\n🎯 Prêt à lancer le pipeline complet!")
    response = input("Continuer? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes', 'oui']:
        print("❌ Annulé par l'utilisateur")
        return 0
    
    # 5. Exécuter le pipeline
    if run_pipeline():
        print(f"\n✅ POC terminé avec succès!")
        print(f"📄 Les résultats sont disponibles dans data/output/")
        return 0
    else:
        print(f"\n❌ POC échoué")
        return 1

if __name__ == "__main__":
    sys.exit(main())