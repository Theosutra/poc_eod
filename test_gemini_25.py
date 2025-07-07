#!/usr/bin/env python3
"""
Test rapide pour vérifier que Gemini 2.5 Flash fonctionne
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

def test_gemini_25_flash():
    """Test simple de Gemini 2.5 Flash"""
    
    # Charger les variables d'environnement
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY manquant dans .env")
        return False
    
    try:
        # Configurer Gemini
        genai.configure(api_key=api_key)
        
        # Créer le modèle Gemini 2.5 Flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Test simple
        print("🧪 Test de Gemini 2.5 Flash...")
        
        prompt = """
        Vous êtes Gemini 2.5 Flash. Confirmez votre version et vos capacités principales en 3 points:
        1. Votre nom et version
        2. Votre vitesse vs Gemini 1.5
        3. Vos capacités vision/OCR
        
        Répondez de manière concise et professionnelle.
        """
        
        response = model.generate_content(prompt)
        
        print("✅ Gemini 2.5 Flash fonctionne parfaitement !")
        print("\n📝 Réponse du modèle:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur avec Gemini 2.5 Flash: {e}")
        print("\n💡 Solutions possibles:")
        print("1. Vérifier que l'API key est valide")
        print("2. Vérifier que gemini-2.5-flash est disponible dans votre région")
        print("3. Essayer avec gemini-1.5-flash en fallback")
        return False

def test_embeddings():
    """Test des embeddings (inchangé)"""
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        genai.configure(api_key=api_key)
        
        print("\n🔗 Test des embeddings...")
        
        result = genai.embed_content(
            model="models/embedding-001",
            content="Test embedding avec Gemini 2.5",
            task_type="retrieval_document"
        )
        
        embedding = result['embedding']
        print(f"✅ Embedding généré: {len(embedding)} dimensions")
        print(f"   Exemple: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur embeddings: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Test de configuration Gemini 2.5 Flash")
    print("=" * 60)
    
    # Test du modèle principal
    success_model = test_gemini_25_flash()
    
    # Test des embeddings
    success_embedding = test_embeddings()
    
    print("\n" + "=" * 60)
    if success_model and success_embedding:
        print("🎉 Configuration Gemini 2.5 Flash : PARFAITE !")
        print("   Le système EODEN est prêt avec la dernière technologie.")
    else:
        print("⚠️  Configuration incomplète - Vérifiez les erreurs ci-dessus")
    
    print("\n📊 Prochaines étapes:")
    print("1. Lancer le pipeline principal: python src/main.py")
    print("2. Tester l'OCR: python src/simplified_embedding_manager.py")
    print("3. Générer des présentations: python src/content_generator.py")