"""
Base vectorielle flexible : Pinecone OU Elasticsearch
Choisir via la config
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class VectorDocument:
    """Document vectorisé"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    score: float = 0.0

class VectorStoreInterface(ABC):
    """Interface commune pour les bases vectorielles"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connexion à la base"""
        pass
    
    @abstractmethod
    def create_index(self, index_name: str, dimension: int) -> bool:
        """Créer un index"""
        pass
    
    @abstractmethod
    def upsert_documents(self, documents: List[VectorDocument], index_name: str) -> bool:
        """Insérer/mettre à jour des documents"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], index_name: str, 
               top_k: int = 10, filters: Dict[str, Any] = None) -> List[VectorDocument]:
        """Recherche par similarité"""
        pass
    
    @abstractmethod
    def delete_index(self, index_name: str) -> bool:
        """Supprimer un index"""
        pass

class PineconeVectorStore(VectorStoreInterface):
    """Implémentation Pinecone avec nouvelle API (basée sur ton exemple)"""
    
    def __init__(self, api_key: str, environment: str = "us-east1-gcp"):
        self.api_key = api_key
        self.environment = environment
        self.pc = None
        self.index = None
        
    def connect(self) -> bool:
        """Connexion à Pinecone avec nouvelle API"""
        try:
            from pinecone import Pinecone
            
            # Nouvelle API Pinecone (comme dans ton exemple)
            self.pc = Pinecone(api_key=self.api_key)
            
            logger.info("✅ Connexion Pinecone réussie")
            return True
            
        except ImportError:
            logger.error("❌ Pinecone non installé: pip install pinecone-client")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur connexion Pinecone: {e}")
            return False
    
    def create_index(self, index_name: str, dimension: int = 768) -> bool:
        """Créer un index Pinecone avec nouvelle API (comme dans ton exemple)"""
        try:
            from pinecone import ServerlessSpec
            
            # Vérifier si l'index existe déjà (comme dans ton exemple)
            try:
                indexes = self.pc.list_indexes()
                index_exists = any(idx.name == index_name for idx in indexes)
            except Exception as e:
                logger.error(f"Erreur lors de la vérification des indexes: {e}")
                return False
            
            if index_exists:
                logger.info(f"Index {index_name} existe déjà")
                self.index = self.pc.Index(index_name)
                return True
            
            # Créer le nouvel index avec ServerlessSpec (gratuit, comme dans ton exemple)
            logger.info(f"Création de l'index {index_name}...")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Attendre que l'index soit prêt (comme dans ton exemple)
            import time
            max_wait = 60  # Timeout de 60 secondes
            wait_time = 0
            while wait_time < max_wait:
                try:
                    indexes = self.pc.list_indexes()
                    if any(idx.name == index_name for idx in indexes):
                        break
                except:
                    pass
                time.sleep(2)
                wait_time += 2
            
            self.index = self.pc.Index(index_name)
            logger.info(f"✅ Index Pinecone créé: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur création index Pinecone: {e}")
            return False
    
    def upsert_documents(self, documents: List[VectorDocument], index_name: str) -> bool:
        """Insérer des documents dans Pinecone (adapté de ton exemple)"""
        try:
            if not self.index:
                self.index = self.pc.Index(index_name)
            
            # Préparer les données pour Pinecone (optimisé pour les métadonnées)
            vectors = []
            for doc in documents:
                # Optimiser les métadonnées pour Pinecone (40KB limit)
                # CORRECTION: Stocker le contenu complet dans les métadonnées
                content_size = len(doc.content)
                if content_size <= 8000:  # Si assez petit, stocker intégralement
                    full_content = doc.content
                    content_preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
                else:  # Sinon, juste aperçu
                    full_content = doc.content[:8000] + "..."
                    content_preview = doc.content[:300] + "..."
                
                metadata = {
                    "source_file": doc.metadata.get("source_file", "")[:200],
                    "document_type": doc.metadata.get("document_type", ""),
                    "section_title": doc.metadata.get("section_title", "")[:100],
                    "business_context": doc.metadata.get("business_context", "")[:150],
                    "themes": doc.metadata.get("themes", [])[:5],
                    "confidence_score": doc.metadata.get("confidence_score", 0.0),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "token_count": doc.metadata.get("token_count", 0),
                    "file_hash": doc.metadata.get("file_hash", ""),
                    "created_at": doc.metadata.get("created_at", ""),
                    # Stocker le contenu complet pour éviter la perte d'information
                    "full_content": full_content,
                    "content_preview": content_preview
                }
                
                vectors.append({
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": metadata
                })
            
            # Upsert par batch de 100 (comme dans ton exemple)
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.debug(f"Batch {i//batch_size + 1} upserted: {len(batch)} vectors")
            
            logger.info(f"✅ {len(documents)} documents insérés dans Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur upsert Pinecone: {e}")
            return False
    
    def search(self, query_embedding: List[float], index_name: str, 
               top_k: int = 10, filters: Dict[str, Any] = None) -> List[VectorDocument]:
        """Recherche dans Pinecone (adapté de ton exemple)"""
        try:
            if not self.index:
                self.index = self.pc.Index(index_name)
            
            # Construire les filtres Pinecone
            pinecone_filter = None
            if filters:
                pinecone_filter = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        pinecone_filter[key] = {"$in": value}
                    else:
                        pinecone_filter[key] = {"$eq": value}
            
            # Recherche (comme dans ton exemple)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Traiter les résultats (adapté de ton exemple de gestion des ScoredVector)
            matches = results.get('matches', [])
            documents = []
            
            for match in matches:
                try:
                    # Gérer les objets ScoredVector ET les dictionnaires (comme dans ton exemple)
                    if hasattr(match, 'score'):
                        # Objet ScoredVector de Pinecone (nouveau format)
                        score = float(match.score)
                        metadata = dict(match.metadata) if hasattr(match, 'metadata') and match.metadata else {}
                        match_id = str(match.id) if hasattr(match, 'id') else ''
                    elif isinstance(match, dict):
                        # Dictionnaire classique (ancien format)
                        score = match.get('score', 0.0)
                        metadata = match.get('metadata', {})
                        match_id = match.get('id', '')
                    else:
                        logger.warning(f"Type de match non reconnu: {type(match)}")
                        continue
                    
                    # CORRECTION: Récupérer le contenu complet en priorité
                    content = metadata.get('full_content', '')
                    if not content:
                        content = metadata.get('content_preview', '')
                    
                    doc = VectorDocument(
                        id=match_id,
                        content=content,
                        embedding=query_embedding,  # Pas retourné par Pinecone
                        metadata=metadata,
                        score=score
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    logger.warning(f"Erreur traitement match: {e}")
                    continue
            
            logger.info(f"✅ Trouvé {len(documents)} résultats Pinecone")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche Pinecone: {e}")
            return []
    
    def delete_index(self, index_name: str) -> bool:
        """Supprimer un index Pinecone"""
        try:
            self.pc.delete_index(index_name)
            logger.info(f"✅ Index Pinecone supprimé: {index_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur suppression index Pinecone: {e}")
            return False
    

class ElasticsearchVectorStore(VectorStoreInterface):
    """Implémentation Elasticsearch"""
    
    def __init__(self, hosts: List[str] = ["localhost:9200"], 
                 username: str = None, password: str = None):
        self.hosts = hosts
        self.username = username
        self.password = password
        self.es = None
    
    def connect(self) -> bool:
        """Connexion à Elasticsearch"""
        try:
            from elasticsearch import Elasticsearch
            
            if self.username and self.password:
                self.es = Elasticsearch(
                    self.hosts,
                    basic_auth=(self.username, self.password),
                    verify_certs=False
                )
            else:
                self.es = Elasticsearch(self.hosts)
            
            # Test de connexion
            if self.es.ping():
                logger.info("✅ Connexion Elasticsearch réussie")
                return True
            else:
                logger.error("❌ Elasticsearch non accessible")
                return False
                
        except ImportError:
            logger.error("❌ Elasticsearch non installé: pip install elasticsearch")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur connexion Elasticsearch: {e}")
            return False
    
    def create_index(self, index_name: str, dimension: int = 768) -> bool:
        """Créer un index Elasticsearch avec mapping vectoriel"""
        try:
            # Mapping pour les vecteurs
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {"type": "object"},
                        "timestamp": {"type": "date"}
                    }
                }
            }
            
            # Créer l'index s'il n'existe pas
            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name, body=mapping)
                logger.info(f"✅ Index Elasticsearch créé: {index_name}")
            else:
                logger.info(f"Index {index_name} existe déjà")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur création index Elasticsearch: {e}")
            return False
    
    def upsert_documents(self, documents: List[VectorDocument], index_name: str) -> bool:
        """Insérer des documents dans Elasticsearch"""
        try:
            from elasticsearch.helpers import bulk
            
            # Préparer les documents
            actions = []
            for doc in documents:
                action = {
                    "_index": index_name,
                    "_id": doc.id,
                    "_source": {
                        "content": doc.content,
                        "embedding": doc.embedding,
                        "metadata": doc.metadata,
                        "timestamp": "now"
                    }
                }
                actions.append(action)
            
            # Insertion en bulk
            success, failed = bulk(self.es, actions)
            logger.info(f"✅ {success} documents insérés dans Elasticsearch")
            
            if failed:
                logger.warning(f"⚠️ {len(failed)} documents échoués")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur upsert Elasticsearch: {e}")
            return False
    
    def search(self, query_embedding: List[float], index_name: str, 
               top_k: int = 10, filters: Dict[str, Any] = None) -> List[VectorDocument]:
        """Recherche dans Elasticsearch"""
        try:
            # Construire la requête
            query = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": top_k * 10
                }
            }
            
            # Ajouter les filtres
            if filters:
                filter_clauses = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {f"metadata.{key}": value}})
                    else:
                        filter_clauses.append({"term": {f"metadata.{key}": value}})
                
                if filter_clauses:
                    query["knn"]["filter"] = {"bool": {"must": filter_clauses}}
            
            # Exécuter la recherche
            response = self.es.search(
                index=index_name,
                body={"query": query},
                size=top_k
            )
            
            # Convertir en VectorDocument
            documents = []
            for hit in response['hits']['hits']:
                doc = VectorDocument(
                    id=hit['_id'],
                    content=hit['_source']['content'],
                    embedding=hit['_source']['embedding'],
                    metadata=hit['_source']['metadata'],
                    score=hit['_score']
                )
                documents.append(doc)
            
            logger.info(f"✅ Trouvé {len(documents)} résultats Elasticsearch")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche Elasticsearch: {e}")
            return []
    
    def delete_index(self, index_name: str) -> bool:
        """Supprimer un index Elasticsearch"""
        try:
            self.es.indices.delete(index=index_name)
            logger.info(f"✅ Index Elasticsearch supprimé: {index_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur suppression index Elasticsearch: {e}")
            return False

class VectorStoreFactory:
    """Factory pour créer la base vectorielle selon la config"""
    
    @staticmethod
    def create_vector_store(store_type: str, config: Dict[str, Any]) -> VectorStoreInterface:
        """Créer une instance de base vectorielle"""
        
        if store_type.lower() == "pinecone":
            return PineconeVectorStore(
                api_key=config.get("api_key"),
                environment=config.get("environment", "us-east1-gcp")
            )
        
        elif store_type.lower() == "elasticsearch":
            return ElasticsearchVectorStore(
                hosts=config.get("hosts", ["localhost:9200"]),
                username=config.get("username"),
                password=config.get("password")
            )
        
        else:
            raise ValueError(f"Type de base vectorielle non supporté: {store_type}")

# Classe principale pour gérer la base vectorielle
class VectorManager:
    """Gestionnaire principal de la base vectorielle"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        vector_config = config.get("vector_store", {})
        store_type = vector_config.get("type", "pinecone")
        store_config = vector_config.get("config", {})
        
        self.store = VectorStoreFactory.create_vector_store(store_type, store_config)
        self.index_name = vector_config.get("index_name", "eoden-documents")
        self.dimension = vector_config.get("dimension", 768)
        
        # Connexion
        if not self.store.connect():
            raise ConnectionError("Impossible de se connecter à la base vectorielle")
        
        # Créer l'index
        self.store.create_index(self.index_name, self.dimension)
    
    def add_documents(self, documents: List[VectorDocument]) -> bool:
        """Ajouter des documents"""
        return self.store.upsert_documents(documents, self.index_name)
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10, 
                      filters: Dict[str, Any] = None) -> List[VectorDocument]:
        """Rechercher des documents similaires"""
        return self.store.search(query_embedding, self.index_name, top_k, filters)
    
    def clear_index(self) -> bool:
        """Vider l'index"""
        return self.store.delete_index(self.index_name) and \
               self.store.create_index(self.index_name, self.dimension)

if __name__ == "__main__":
    # Test basique
    print("🧪 Test des bases vectorielles...")
    
    # Exemple de config
    test_config = {
        "vector_store": {
            "type": "pinecone",  # ou "elasticsearch"
            "index_name": "test-index",
            "dimension": 768,
            "config": {
                "api_key": "your-pinecone-key",
                "environment": "us-east1-gcp"
            }
        }
    }
    
    print("✅ Structure prête ! Configurer settings.yaml et tester.")