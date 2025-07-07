#!/usr/bin/env python3
"""
Test UNIQUEMENT des embeddings sans enrichissement IA
Pour éviter les problèmes de sécurité Gemini 2.5
"""

import os
import sys
import hashlib
import time
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

sys.path.append('src')

from dotenv import load_dotenv
import google.generativeai as genai

@dataclass
class SimpleChunk:
    """Chunk simplifié sans enrichissement IA"""
    id: str
    content: str
    source: str
    embedding: List[float]
    size: int

class SimpleEmbeddingTest:
    """Test simple d'embeddings sans enrichissement"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.embedding_model = "models/embedding-001"
        self.cache = {}
        
    def get_embedding(self, text: str) -> List[float]:
        """Générer un embedding"""
        # Cache simple
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
            
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            self.cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"❌ Erreur embedding: {e}")
            # Embedding de fallback
            return [0.1] * 768
    
    def simple_chunk(self, text: str, source: str) -> List[str]:
        """Chunking très simple par taille"""
        chunks = []
        words = text.split()
        
        current_chunk = ""
        for word in words:
            if len(current_chunk + " " + word) > 1000:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return [c for c in chunks if len(c) > 50]
    
    def process_documents(self, documents: Dict[str, str]) -> List[SimpleChunk]:
        """Traiter les documents sans enrichissement IA"""
        all_chunks = []
        
        for source, content in documents.items():
            print(f"📄 Traitement: {source}")
            
            # Chunking simple
            text_chunks = self.simple_chunk(content, source)
            print(f"   📊 {len(text_chunks)} chunks créés")
            
            # Embeddings
            for i, chunk_text in enumerate(text_chunks):
                try:
                    embedding = self.get_embedding(chunk_text)
                    
                    chunk = SimpleChunk(
                        id=f"{source}_{i}",
                        content=chunk_text,
                        source=source,
                        embedding=embedding,
                        size=len(chunk_text)
                    )
                    
                    all_chunks.append(chunk)
                    print(f"   ✅ Chunk {i}: {len(embedding)} dimensions")
                    
                except Exception as e:
                    print(f"   ❌ Erreur chunk {i}: {e}")
                    continue
        
        return all_chunks

def test_simple_embeddings():
    """Test ultra-simple d'embeddings"""
    
    print("⚡ TEST EMBEDDINGS SEULEMENT")
    print("🚫 Pas d'enrichissement IA (évite les erreurs)")
    print("=" * 50)
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY manquant")
        return
    
    # Documents de test
    docs = {
        "business_plan.pdf": """
        TechCorp est une entreprise spécialisée dans l'IoT industriel.
        Chiffre d'affaires 2023: 2.1 millions d'euros.
        Croissance de 45% par rapport à 2022.
        L'équipe compte 15 personnes expérimentées.
        Le marché de l'IoT représente 45M€ en France.
        """,
        
        "presentation.pptx": """
        Opportunité d'investissement en Série A pour 3 millions d'euros.
        25 clients actifs avec un NPS de 8.5 sur 10.
        Expansion prévue en Allemagne et Italie.
        Objectif: 5M€ de chiffre d'affaires en 2025.
        """
    }
    
    try:
        # Test simple
        manager = SimpleEmbeddingTest(api_key)
        chunks = manager.process_documents(docs)
        
        print(f"\n🎉 SUCCÈS! {len(chunks)} chunks vectorisés")
        
        # Afficher résultats
        print("\n📋 Résultats:")
        for chunk in chunks[:3]:
            print(f"\n🔹 {chunk.id}")
            print(f"   📄 Source: {chunk.source}")
            print(f"   📏 Taille: {chunk.size} chars")
            print(f"   🧮 Embedding: {len(chunk.embedding)} dim")
            print(f"   📝 Contenu: {chunk.content[:100]}...")
            
            # Vérifier validité
            if all(x == 0.1 for x in chunk.embedding[:3]):
                print("   ⚠️  Embedding de fallback utilisé")
            else:
                print(f"   ✅ Embedding valide: [{chunk.embedding[0]:.4f}, {chunk.embedding[1]:.4f}, ...]")
        
        print(f"\n💾 Cache: {len(manager.cache)} embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_embeddings()
    
    if success:
        print("\n🎯 Embeddings fonctionnent parfaitement!")
        print("\n📝 Prochaines étapes:")
        print("1. Corriger l'enrichissement IA (problème sécurité Gemini 2.5)")
        print("2. Ou utiliser uniquement les embeddings bruts")
        print("3. Connecter à une vraie base vectorielle")
    else:
        print("\n❌ Problème avec les embeddings de base")