"""
Gestionnaire des embeddings avec Gemini
Chunking intelligent + vectorisation
"""

import os
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
import numpy as np
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Types de documents supportés"""
    BUSINESS_PLAN = "business_plan"
    FINANCIAL_AUDIT = "financial_audit"
    MARKET_STUDY = "market_study"
    LEGAL_AUDIT = "legal_audit"
    PRESENTATION = "presentation"
    EXCEL_REPORT = "excel_report"
    UNKNOWN = "unknown"

@dataclass
class ChunkMetadata:
    """Métadonnées d'un chunk"""
    source_file: str
    document_type: DocumentType
    section_title: str
    page_number: int
    chunk_index: int
    business_context: str
    themes: List[str]
    confidence_score: float
    file_hash: str
    created_at: str

@dataclass
class ProcessedChunk:
    """Chunk traité avec embedding"""
    id: str
    content: str
    embedding: List[float]
    metadata: ChunkMetadata
    token_count: int

class EmbeddingManager:
    """Gestionnaire des embeddings avec Gemini"""
    
    def __init__(self, api_key: str, cache_dir: str = "data/embeddings"):
        genai.configure(api_key=api_key)
        self.embedding_model = "models/embedding-001"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration du chunking
        self.max_tokens = 1800  # Sécurité sous les 2048 de Gemini
        self.overlap_tokens = 200
        self.rate_limit_delay = 0.1  # Délai entre les requêtes
        
        # Cache des embeddings
        self.embedding_cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Charger le cache des embeddings"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Cache chargé: {len(self.embedding_cache)} embeddings")
            except Exception as e:
                logger.warning(f"Erreur chargement cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Sauvegarder le cache des embeddings"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Cache sauvegardé: {len(self.embedding_cache)} embeddings")
        except Exception as e:
            logger.error(f"Erreur sauvegarde cache: {e}")
    
    def detect_document_type(self, filename: str, content: str) -> DocumentType:
        """Détecter le type de document"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        # Détection par nom de fichier
        if any(word in filename_lower for word in ['business_plan', 'bp', 'plan_affaires']):
            return DocumentType.BUSINESS_PLAN
        elif any(word in filename_lower for word in ['audit_financier', 'finance', 'comptable']):
            return DocumentType.FINANCIAL_AUDIT
        elif any(word in filename_lower for word in ['marche', 'market', 'etude']):
            return DocumentType.MARKET_STUDY
        elif any(word in filename_lower for word in ['juridique', 'legal', 'droit']):
            return DocumentType.LEGAL_AUDIT
        elif any(word in filename_lower for word in ['presentation', 'ppt', 'slides']):
            return DocumentType.PRESENTATION
        
        # Détection par contenu
        if any(word in content_lower for word in ['chiffre d\'affaires', 'ebitda', 'bilan']):
            return DocumentType.FINANCIAL_AUDIT
        elif any(word in content_lower for word in ['marché', 'concurrence', 'segment']):
            return DocumentType.MARKET_STUDY
        elif any(word in content_lower for word in ['business model', 'stratégie', 'équipe']):
            return DocumentType.BUSINESS_PLAN
        
        return DocumentType.UNKNOWN
    
    def get_chunking_strategy(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Stratégie de chunking par type de document"""
        strategies = {
            DocumentType.BUSINESS_PLAN: {
                "chunk_size": 1500,
                "overlap": 200,
                "split_markers": ["##", "###", "Résumé", "Analyse", "Stratégie"],
                "preserve_context": True
            },
            DocumentType.FINANCIAL_AUDIT: {
                "chunk_size": 1200,
                "overlap": 150,
                "split_markers": ["Tableau", "Analyse", "Période", "Exercice"],
                "preserve_context": True
            },
            DocumentType.PRESENTATION: {
                "chunk_size": 800,
                "overlap": 100,
                "split_markers": ["Slide", "Diapositive", "---"],
                "preserve_context": False
            },
            DocumentType.MARKET_STUDY: {
                "chunk_size": 1600,
                "overlap": 250,
                "split_markers": ["Marché", "Segment", "Tendance", "Analyse"],
                "preserve_context": True
            },
            DocumentType.LEGAL_AUDIT: {
                "chunk_size": 1400,
                "overlap": 200,
                "split_markers": ["Article", "Clause", "Risque", "Conformité"],
                "preserve_context": True
            },
            DocumentType.EXCEL_REPORT: {
                "chunk_size": 1000,
                "overlap": 100,
                "split_markers": ["Feuille", "Tableau", "Graphique"],
                "preserve_context": True
            },
            DocumentType.UNKNOWN: {
                "chunk_size": 1400,
                "overlap": 200,
                "split_markers": ["###", "\n\n"],
                "preserve_context": True
            }
        }
        return strategies[doc_type]
    
    def intelligent_chunking(self, text: str, doc_type: DocumentType, 
                           source_file: str) -> List[str]:
        """Chunking intelligent basé sur le type de document"""
        strategy = self.get_chunking_strategy(doc_type)
        
        # Préprocessing du texte
        text = self._preprocess_text(text, doc_type)
        
        # CORRECTION: Limiter la taille des chunks pour éviter l'erreur 36KB
        max_chunk_size = min(strategy["chunk_size"], 8000)  # Max 8000 chars pour sécurité
        
        # Chunking hiérarchique
        chunks = self._hierarchical_split(
            text, 
            strategy["split_markers"], 
            max_chunk_size,
            strategy["overlap"]
        )
        
        # Filtrer les chunks trop petits ET trop gros
        filtered_chunks = []
        for chunk in chunks:
            chunk_size = len(chunk.strip())
            if 100 < chunk_size < 10000:  # Entre 100 et 10k caractères
                filtered_chunks.append(chunk)
            elif chunk_size >= 10000:
                # Si chunk trop gros, le diviser en plus petits
                sub_chunks = self._force_split_large_chunk(chunk, 8000)
                filtered_chunks.extend(sub_chunks)
        
        logger.info(f"Document {source_file}: {len(filtered_chunks)} chunks créés")
        return filtered_chunks
    
    def _force_split_large_chunk(self, chunk: str, max_size: int) -> List[str]:
        """Forcer la division d'un chunk trop volumineux"""
        sub_chunks = []
        words = chunk.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk + " " + word) > max_size:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        
        if current_chunk:
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks
    
    def _preprocess_text(self, text: str, doc_type: DocumentType) -> str:
        """Préprocessing spécifique par type"""
        import re
        
        # Nettoyage général
        text = re.sub(r'\s+', ' ', text)  # Espaces multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Sauts de ligne multiples
        
        if doc_type == DocumentType.FINANCIAL_AUDIT:
            # Préserver les formats numériques
            text = re.sub(r'(\d+[.,]\d+)\s*([€$M%])', r'\1 \2', text)
        
        elif doc_type == DocumentType.PRESENTATION:
            # Structurer les slides
            text = re.sub(r'^([A-Z][^.!?]*?)$', r'## \1', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _hierarchical_split(self, text: str, split_markers: List[str], 
                          max_size: int, overlap: int) -> List[str]:
        """Split hiérarchique respectant la structure"""
        chunks = []
        current_chunk = ""
        
        lines = text.split('\n')
        
        for line in lines:
            # Vérifier les marqueurs de section
            is_section_break = any(marker.lower() in line.lower() for marker in split_markers)
            
            if is_section_break and current_chunk and self._count_tokens(current_chunk) > 200:
                # Finaliser le chunk précédent
                chunks.append(current_chunk.strip())
                
                # Commencer nouveau chunk avec overlap
                if overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + "\n" + line + '\n'
                else:
                    current_chunk = line + '\n'
            else:
                # Ajouter au chunk actuel
                potential_chunk = current_chunk + line + '\n'
                
                if self._count_tokens(potential_chunk) > max_size:
                    # Chunk trop grand, finaliser
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    # Nouveau chunk avec overlap
                    if overlap > 0:
                        overlap_text = self._get_overlap_text(current_chunk, overlap)
                        current_chunk = overlap_text + "\n" + line + '\n'
                    else:
                        current_chunk = line + '\n'
                else:
                    current_chunk = potential_chunk
        
        # Ajouter le dernier chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Extraire le texte de chevauchement"""
        words = text.split()
        # Approximation: 1 token ≈ 0.75 mots en français
        overlap_words = int(overlap_tokens * 0.75)
        return ' '.join(words[-overlap_words:]) if len(words) > overlap_words else text
    
    def _count_tokens(self, text: str) -> int:
        """Comptage approximatif des tokens"""
        # Approximation pour le français: 1 token ≈ 4 caractères
        return len(text) // 4
    
    def generate_metadata(self, chunk: str, doc_type: DocumentType, 
                         source_file: str, chunk_index: int) -> ChunkMetadata:
        """Générer les métadonnées d'un chunk avec Gemini"""
        
        # Créer un hash du fichier source
        file_hash = hashlib.md5(source_file.encode()).hexdigest()[:8]
        
        # Limiter la taille du chunk pour le prompt (max 1000 chars)
        chunk_sample = chunk[:1000] if len(chunk) > 1000 else chunk
        
        # Prompt simplifié pour extraction des métadonnées
        metadata_prompt = f"""
        Analysez ce document et répondez en JSON:
        
        Type: {doc_type.value}
        Texte: {chunk_sample}
        
        JSON:
        {{
            "section_title": "titre section",
            "business_context": "contexte business",
            "themes": ["thème1", "thème2"],
            "confidence_score": 0.8
        }}
        """
        
        try:
            # CORRECTION: Utiliser le bon modèle Gemini
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                metadata_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )
            )
            
            # DEBUG: Log de la réponse brute
            logger.debug(f"Réponse brute Gemini: {response.text}")
            
            # Parser le JSON avec gestion d'erreur améliorée
            import json
            import re
            
            response_text = response.text.strip()
            
            # Chercher le JSON dans la réponse (parfois il y a du texte avant/après)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                metadata_dict = json.loads(json_str)
            else:
                # Si pas de JSON trouvé, utiliser les valeurs par défaut
                logger.warning(f"Pas de JSON trouvé dans la réponse: {response_text}")
                raise ValueError("Pas de JSON valide dans la réponse")
            
            return ChunkMetadata(
                source_file=source_file,
                document_type=doc_type,
                section_title=metadata_dict.get("section_title", "Unknown")[:50],
                page_number=0,  # On ne peut pas deviner facilement
                chunk_index=chunk_index,
                business_context=metadata_dict.get("business_context", ""),
                themes=metadata_dict.get("themes", [])[:3],
                confidence_score=metadata_dict.get("confidence_score", 0.5),
                file_hash=file_hash,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            logger.warning(f"Erreur génération métadonnées: {e}")
            # Créer des métadonnées basiques à partir du contenu
            return self._create_fallback_metadata(chunk, doc_type, source_file, chunk_index, file_hash)
    
    def _create_fallback_metadata(self, chunk: str, doc_type: DocumentType, 
                                 source_file: str, chunk_index: int, file_hash: str) -> ChunkMetadata:
        """Créer des métadonnées de fallback sans API"""
        # Extraire le titre de section simple
        lines = chunk.split('\n')
        section_title = "Unknown"
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('#') or line.isupper() or len(line) < 60):
                section_title = line.replace('#', '').strip()[:50]
                break
        
        # Thèmes basiques selon le type de document
        themes = []
        if doc_type == DocumentType.FINANCIAL_AUDIT:
            themes = ["finance", "audit"]
        elif doc_type == DocumentType.BUSINESS_PLAN:
            themes = ["business", "stratégie"]
        elif doc_type == DocumentType.MARKET_STUDY:
            themes = ["marché", "analyse"]
        elif doc_type == DocumentType.PRESENTATION:
            themes = ["présentation"]
        else:
            themes = ["document"]
        
        return ChunkMetadata(
            source_file=source_file,
            document_type=doc_type,
            section_title=section_title,
            page_number=0,
            chunk_index=chunk_index,
            business_context=f"Contexte {doc_type.value}",
            themes=themes,
            confidence_score=0.3,
            file_hash=file_hash,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Générer un embedding avec cache et validation"""
        # Créer une clé de cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Vérifier le cache
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # CORRECTION: Limiter la taille du texte pour éviter l'erreur
        if len(text) > 20000:  # Limite sécurisée
            text = text[:20000] + "..."
            logger.warning(f"Texte tronqué pour embedding: {len(text)} chars")
        
        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)
            
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            
            # VALIDATION: Vérifier que l'embedding n'est pas vide/zéro
            if not embedding or all(x == 0.0 for x in embedding):
                logger.error("Embedding vide généré, utilisation d'un embedding par défaut")
                # Générer un embedding aléatoire minimal mais valide
                import random
                embedding = [random.uniform(-0.1, 0.1) for _ in range(768)]
            
            # Mettre en cache
            self.embedding_cache[text_hash] = embedding
            
            # Sauvegarder le cache périodiquement
            if len(self.embedding_cache) % 50 == 0:  # Plus fréquent
                self._save_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur embedding: {e}")
            # CORRECTION: Retourner un embedding aléatoire valide au lieu de zéros
            import random
            fallback_embedding = [random.uniform(-0.1, 0.1) for _ in range(768)]
            logger.warning("Utilisation d'un embedding de fallback")
            return fallback_embedding
    
    def process_document(self, file_path: str, content: str) -> List[ProcessedChunk]:
        """Traiter un document complet: chunking + embedding + métadonnées"""
        
        # Détecter le type de document
        doc_type = self.detect_document_type(file_path, content)
        logger.info(f"Document {file_path} détecté comme: {doc_type.value}")
        
        # Chunking intelligent
        chunks = self.intelligent_chunking(content, doc_type, file_path)
        
        # Traiter chaque chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Générer les métadonnées
                metadata = self.generate_metadata(chunk, doc_type, file_path, i)
                
                # Générer l'embedding
                embedding = self.get_embedding(chunk)
                
                # Créer l'ID unique
                chunk_id = f"{metadata.file_hash}_{i:03d}"
                
                # Créer le chunk traité
                processed_chunk = ProcessedChunk(
                    id=chunk_id,
                    content=chunk,
                    embedding=embedding,
                    metadata=metadata,
                    token_count=self._count_tokens(chunk)
                )
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.error(f"Erreur traitement chunk {i}: {e}")
                continue
        
        logger.info(f"Document traité: {len(processed_chunks)} chunks")
        
        # Sauvegarder le cache final
        self._save_cache()
        
        return processed_chunks
    
    def batch_process_documents(self, file_contents: Dict[str, str]) -> List[ProcessedChunk]:
        """Traiter plusieurs documents en batch"""
        all_chunks = []
        
        logger.info(f"Traitement de {len(file_contents)} documents...")
        
        for file_path, content in file_contents.items():
            try:
                chunks = self.process_document(file_path, content)
                all_chunks.extend(chunks)
                logger.info(f"✅ {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"❌ Erreur traitement {file_path}: {e}")
        
        logger.info(f"✅ Traitement terminé: {len(all_chunks)} chunks au total")
        return all_chunks
    
    def search_embeddings(self, query: str, chunks: List[ProcessedChunk], 
                         top_k: int = 10) -> List[ProcessedChunk]:
        """Recherche par similarité dans les chunks"""
        
        # Générer l'embedding de la requête
        query_embedding = self.get_embedding(query)
        
        # Calculer les similarités
        similarities = []
        for chunk in chunks:
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            similarities.append((chunk, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les top_k
        return [chunk for chunk, _ in similarities[:top_k]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculer la similarité cosinus"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def clear_cache(self):
        """Vider le cache des embeddings"""
        self.embedding_cache = {}
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Cache des embeddings vidé")

# Fonction de test
def test_embedding_manager():
    """Test du gestionnaire d'embeddings"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY manquant dans .env")
        return
    
    # Créer le gestionnaire
    manager = EmbeddingManager(api_key)
    
    # Test avec un texte simple
    test_content = """
    ## Résumé Exécutif
    
    TechCorp est une startup innovante spécialisée dans l'IoT industriel.
    Le marché cible est estimé à 45M€ avec une croissance de 15% par an.
    
    ## Analyse Financière
    
    Chiffre d'affaires 2023: 2.1M€
    EBITDA: 18%
    Projection 2024: 3.2M€
    
    ## Équipe de Direction
    
    CEO: Jean Dupont (10 ans d'expérience IoT)
    CTO: Marie Martin (Expert en systèmes embarqués)
    """
    
    print("🧪 Test du traitement de document...")
    chunks = manager.process_document("test_business_plan.pdf", test_content)
    
    print(f"✅ {len(chunks)} chunks créés")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Contenu: {chunk.content[:100]}...")
        print(f"  Métadonnées: {chunk.metadata.section_title}")
        print(f"  Thèmes: {chunk.metadata.themes}")
        print(f"  Embedding: {len(chunk.embedding)} dimensions")
    
    # Test de recherche
    print("\n🔍 Test de recherche...")
    results = manager.search_embeddings("analyse financière", chunks, top_k=2)
    
    for i, result in enumerate(results):
        print(f"\nRésultat {i+1}:")
        print(f"  Section: {result.metadata.section_title}")
        print(f"  Contenu: {result.content[:150]}...")

if __name__ == "__main__":
    test_embedding_manager()