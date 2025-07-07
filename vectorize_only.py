#!/usr/bin/env python3
"""
Script pour vectorisation rapide SANS OCR
Test uniquement le chunking + enrichissement + embeddings
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ajouter le dossier src au path
sys.path.append('src')

from simplified_embedding_manager import SimplifiedEmbeddingManager

# Configuration logging simple
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def vectorize_test_documents():
    """Test de vectorisation sur des documents simulés"""
    
    print("🚀 Vectorisation rapide SANS OCR")
    print("=" * 50)
    
    # Charger les variables d'environnement
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY manquant dans .env")
        return False
    
    try:
        # Créer le gestionnaire simplifié
        print("🧠 Initialisation du gestionnaire d'embeddings...")
        manager = SimplifiedEmbeddingManager(api_key)
        
        # Documents de test (simulés)
        test_documents = {
            "business_plan_techcorp.pdf": """
            ## Présentation de TechCorp
            
            TechCorp est une startup française spécialisée dans l'IoT industriel.
            L'entreprise a été fondée en 2020 par Jean Dupont et Marie Martin.
            Elle développe des solutions connectées pour l'industrie 4.0.
            
            ## Chiffres Clés 2023
            
            Chiffre d'affaires: 2.1M€ (+45% vs 2022)
            EBITDA: 18%
            Effectifs: 15 personnes
            Clients: 25 entreprises industrielles
            
            ## Équipe de Direction
            
            Jean Dupont, CEO: 10 ans d'expérience en IoT, ancien directeur technique chez Schneider Electric
            Marie Martin, CTO: Experte en systèmes embarqués, diplômée Centrale Paris
            
            ## Marché et Concurrence
            
            Le marché de l'IoT industriel représente 45M€ en France avec une croissance de 15% par an.
            Les principaux concurrents sont Sigfox, Orange Business et des startups comme Objenious.
            
            ## Stratégie de Développement
            
            L'entreprise vise 5M€ de CA en 2025 grâce à:
            - Expansion internationale (Allemagne, Italie)
            - Nouveaux produits (capteurs environnementaux)
            - Partenariats industriels stratégiques
            """,
            
            "presentation_investissement.pptx": """
            --- Slide 1 ---
            TechCorp - Opportunité d'Investissement
            Série A - 3M€
            
            --- Slide 2 ---
            Problème Adressé
            • 70% des usines manquent de données temps réel
            • Maintenance curative coûteuse
            • Gaspillage énergétique important
            
            --- Slide 3 ---
            Notre Solution
            • Capteurs IoT plug-and-play
            • Plateforme analytics cloud
            • ROI prouvé de 6 mois
            
            --- Slide 4 ---
            Traction Commerciale
            • 25 clients actifs
            • 2.1M€ de CA en 2023
            • Croissance +45% YoY
            • NPS Score: 8.5/10
            
            --- Slide 5 ---
            Utilisation des Fonds
            • R&D (40%): Nouveaux capteurs
            • Sales & Marketing (35%): Expansion
            • Recrutement (25%): 10 personnes
            """,
            
            "analyse_financiere_2023.xlsx": """
            --- Feuille: Compte de Résultat ---
            Exercice 2023 | Exercice 2022 | Évolution
            CA | 2,100,000€ | 1,450,000€ | +45%
            Charges | 1,722,000€ | 1,276,000€ | +35%
            EBITDA | 378,000€ | 174,000€ | +117%
            Résultat Net | 252,000€ | 87,000€ | +190%
            
            --- Feuille: Métriques Clés ---
            Marge EBITDA | 18% | 12% | +6pts
            Marge Nette | 12% | 6% | +6pts
            CA par employé | 140,000€ | 121,000€ | +16%
            Croissance MRR | 8% | 5% | +3pts
            
            --- Feuille: Projections 2024 ---
            CA Objectif | 3,200,000€
            EBITDA Cible | 22%
            Nouveaux Clients | 15
            Expansion Géo | 2 pays
            """
        }
        
        print(f"\n📄 Traitement de {len(test_documents)} documents de test...")
        
        # Vectoriser les documents
        all_chunks = manager.batch_process_documents(test_documents)
        
        print(f"\n✅ Vectorisation terminée!")
        print(f"   📊 {len(all_chunks)} chunks créés au total")
        
        # Afficher les métriques
        print("\n" + "="*50)
        manager.print_metrics()
        
        # Afficher quelques exemples de chunks
        print("\n📋 Exemples de chunks enrichis:")
        print("-" * 40)
        
        for i, chunk in enumerate(all_chunks[:5], 1):
            print(f"\n🔹 Chunk {i}:")
            print(f"   📄 Document: {chunk.metadata.nom_document}")
            print(f"   🎯 Thème: {chunk.enriched_content.theme_principal}")
            print(f"   💬 Guide: {chunk.enriched_content.commentaire_guidage}")
            print(f"   🔗 Vectorisé: {chunk.enriched_content.contenu_enrichi}")
            print(f"   📏 Taille: {chunk.metadata.taille_chunk} chars")
            print(f"   🧮 Embedding: {len(chunk.embedding)} dimensions")
            
            # Vérifier que l'embedding n'est pas vide
            if all(x == 0.0 for x in chunk.embedding[:10]):
                print("   ⚠️  ATTENTION: Embedding semble vide!")
            else:
                print(f"   ✅ Embedding valide: [{chunk.embedding[0]:.4f}, {chunk.embedding[1]:.4f}, ...]")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la vectorisation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_search_similarity():
    """Test rapide de recherche par similarité"""
    
    print("\n" + "="*50)
    print("🔍 Test de recherche par similarité")
    
    # Ici on pourrait tester la recherche si on avait une base vectorielle
    # Pour l'instant, on confirme juste que la vectorisation fonctionne
    print("   (Implémentation recherche dans vector_store.py)")

if __name__ == "__main__":
    print("🧪 Test de Vectorisation Rapide - EODEN POC")
    print("   Mode: SANS OCR pour maximum de vitesse")
    print("   Objectif: Tester chunking + enrichissement + embeddings")
    
    success = vectorize_test_documents()
    
    if success:
        test_search_similarity()
        print("\n🎉 Test de vectorisation : RÉUSSI!")
        print("\n📝 Prochaines étapes:")
        print("1. Activer l'OCR si nécessaire (config/settings.yaml)")
        print("2. Connecter à Google Drive pour vrais documents")
        print("3. Configurer la base vectorielle (Pinecone/Elasticsearch)")
    else:
        print("\n❌ Test échoué - Vérifiez la configuration")
        
    print("\n💡 Pour désactiver complètement l'OCR:")
    print("   Éditez config/settings.yaml → ocr: enabled: false")