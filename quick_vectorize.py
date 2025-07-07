#!/usr/bin/env python3
"""
Vectorisation ULTRA-RAPIDE sans OCR
Mode test pour vérifier le pipeline
"""

import os
import sys
from pathlib import Path

# Configuration pour désactiver l'OCR temporairement
os.environ['EODEN_NO_OCR'] = 'true'

# Ajouter src au path
sys.path.append('src')

from dotenv import load_dotenv
from simplified_embedding_manager import SimplifiedEmbeddingManager

def quick_test():
    """Test ultra-rapide avec documents simulés"""
    
    print("⚡ VECTORISATION RAPIDE - Mode Test")
    print("🚫 OCR désactivé pour maximum de vitesse")
    print("=" * 50)
    
    # Charger config
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Manque GEMINI_API_KEY dans .env")
        return
    
    # Documents de test courts
    docs = {
        "test1.pdf": """
        Présentation de l'Entreprise TechCorp
        
        TechCorp développe des solutions IoT pour l'industrie.
        Chiffre d'affaires 2023: 2.1M€
        Croissance: +45%
        Équipe: 15 personnes
        """,
        
        "test2.pptx": """
        --- Slide 1 ---
        Opportunité d'Investissement
        Série A: 3M€
        
        --- Slide 2 ---
        Traction Commerciale
        25 clients actifs
        NPS: 8.5/10
        """
    }
    
    try:
        print("🧠 Initialisation...")
        manager = SimplifiedEmbeddingManager(api_key)
        
        print("📄 Traitement documents...")
        chunks = manager.batch_process_documents(docs)
        
        print(f"\n✅ SUCCÈS! {len(chunks)} chunks créés")
        
        # Afficher résultats
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n📋 Chunk {i}:")
            print(f"   🎯 {chunk.enriched_content.theme_principal}")
            print(f"   💬 {chunk.enriched_content.commentaire_guidage}")
            print(f"   📊 Embedding: {len(chunk.embedding)} dim")
        
        # Métriques
        print("\n📊 Métriques:")
        manager.print_metrics()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()