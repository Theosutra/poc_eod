"""
Gestionnaire d'embeddings simplifié selon votre vision
Métadonnées basiques + Contenu vectorisé enrichi par IA
Fichier: src/simplified_embedding_manager.py
"""

import os
import logging
import hashlib
import time
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import google.generativeai as genai

logger = logging.getLogger(__name__)

@dataclass
class SimplifiedMetadata:
    """Métadonnées simplifiées - structure du fichier"""
    contenu_chunk: str          # Texte brut du chunk
    date_creation: str          # Timestamp de création
    nom_document: str           # Nom du fichier source
    dossier_racine: str         # Dossier parent dans Drive
    chunk_index: int            # Index du chunk dans le document
    taille_chunk: int           # Nombre de caractères

@dataclass
class EnrichedContent:
    """Contenu enrichi pour la vectorisation"""
    theme_principal: str        # Ex: "Présentation de l'entreprise"
    commentaire_guidage: str    # Description courte pour aiguiller
    contenu_enrichi: str        # Texte final à vectoriser

@dataclass
class SimplifiedChunk:
    """Chunk final avec métadonnées + contenu enrichi"""
    id: str
    metadata: SimplifiedMetadata
    enriched_content: EnrichedContent
    embedding: List[float]

class SimplifiedEmbeddingManager:
    """Gestionnaire d'embeddings avec approche simplifiée"""
    
    def __init__(self, api_key: str, cache_dir: str = "data/embeddings"):
        if not api_key:
            raise ValueError("API key is required")
        
        genai.configure(api_key=api_key)
        self.embedding_model = "models/embedding-001"
        self.generation_model = genai.GenerativeModel('gemini-2.5-flash')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache des enrichissements
        self.enrichment_cache = {}
        self.embedding_cache = {}
        
        # Métriques de performance
        self.metrics = {
            "embeddings_generated": 0,
            "enrichments_generated": 0,
            "cache_hits_embedding": 0,
            "cache_hits_enrichment": 0
        }
        
        self._load_caches()
    
    def _load_caches(self):
        """Charger les caches d'enrichissement et d'embeddings"""
        
        # Cache enrichissement
        enrichment_file = self.cache_dir / "enrichment_cache.json"
        if enrichment_file.exists():
            try:
                with open(enrichment_file, 'r', encoding='utf-8') as f:
                    self.enrichment_cache = json.load(f)
                logger.info(f"Cache enrichissement chargé: {len(self.enrichment_cache)} entrées")
            except Exception as e:
                logger.warning(f"Erreur chargement cache enrichissement: {e}")
                self.enrichment_cache = {}
        
        # Cache embeddings
        embedding_file = self.cache_dir / "embedding_cache.json"
        if embedding_file.exists():
            try:
                with open(embedding_file, 'r', encoding='utf-8') as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"Cache embeddings chargé: {len(self.embedding_cache)} entrées")
            except Exception as e:
                logger.warning(f"Erreur chargement cache embeddings: {e}")
                self.embedding_cache = {}
    
    def _save_caches(self):
        """Sauvegarder les caches"""
        
        # Cache enrichissement
        try:
            enrichment_file = self.cache_dir / "enrichment_cache.json"
            with open(enrichment_file, 'w', encoding='utf-8') as f:
                json.dump(self.enrichment_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache enrichissement sauvegardé: {len(self.enrichment_cache)} entrées")
        except Exception as e:
            logger.error(f"Erreur sauvegarde cache enrichissement: {e}")
        
        # Cache embeddings
        try:
            embedding_file = self.cache_dir / "embedding_cache.json"
            with open(embedding_file, 'w', encoding='utf-8') as f:
                json.dump(self.embedding_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Cache embeddings sauvegardé: {len(self.embedding_cache)} entrées")
        except Exception as e:
            logger.error(f"Erreur sauvegarde cache embeddings: {e}")
    
    def enrich_chunk_content(self, chunk_text: str, document_name: str) -> EnrichedContent:
        """Enrichir le contenu d'un chunk avec l'IA"""
        
        # Vérifier le cache
        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
        if chunk_hash in self.enrichment_cache:
            self.metrics["cache_hits_enrichment"] += 1
            cached = self.enrichment_cache[chunk_hash]
            return EnrichedContent(
                theme_principal=cached["theme_principal"],
                commentaire_guidage=cached["commentaire_guidage"],
                contenu_enrichi=cached["contenu_enrichi"]
            )
        
        # Prompt simplifié pour l'enrichissement
        prompt = f"""
Vous êtes un expert en analyse de documents d'investissement.

ANALYSEZ ce chunk de texte et définissez:

DOCUMENT SOURCE: {document_name}
TEXTE À ANALYSER:
{chunk_text[:1500]}

RÉPONDEZ EN JSON:
{{
    "theme_principal": "Catégorie principale",
    "commentaire_guidage": "Description courte en 1-2 phrases pour aiguiller la recherche"
}}

THÈMES POSSIBLES:
- Présentation de l'entreprise
- Analyse financière
- Étude de marché
- Équipe de direction
- Stratégie et business model
- Risques et opportunités
- Aspects juridiques
- Performance opérationnelle
- Audit technique
- Gouvernance et organisation
- Projets et développement
- Partenariats et alliances

EXEMPLES:
• Chunk sur CA/EBITDA → "Analyse financière" : "Indicateurs de performance et rentabilité"
• Chunk sur CEO/équipe → "Équipe de direction" : "Composition et expérience de l'équipe dirigeante"
• Chunk sur marché/concurrence → "Étude de marché" : "Analyse du secteur et positionnement concurrentiel"
"""
        
        try:
            # Générer l'enrichissement avec config sécurité
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=300
                ),
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Vérifier si la réponse est valide
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning(f"Réponse Gemini vide ou bloquée. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}")
                raise ValueError("Réponse Gemini bloquée par les filtres de sécurité")
            
            # Parser la réponse JSON
            response_text = response.text.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
                
                theme_principal = result.get("theme_principal", "Contenu général")
                commentaire_guidage = result.get("commentaire_guidage", "Informations diverses")
                
                # Créer le contenu enrichi pour vectorisation
                contenu_enrichi = f"{theme_principal} : {commentaire_guidage}"
                
                # Mettre en cache
                self.enrichment_cache[chunk_hash] = {
                    "theme_principal": theme_principal,
                    "commentaire_guidage": commentaire_guidage,
                    "contenu_enrichi": contenu_enrichi
                }
                
                # Mettre à jour les métriques
                self.metrics["enrichments_generated"] += 1
                
                # Sauvegarder le cache périodiquement
                if len(self.enrichment_cache) % 10 == 0:
                    self._save_caches()
                
                logger.info(f"Enrichissement: {theme_principal} - {commentaire_guidage[:50]}...")
                
                return EnrichedContent(
                    theme_principal=theme_principal,
                    commentaire_guidage=commentaire_guidage,
                    contenu_enrichi=contenu_enrichi
                )
            
            else:
                raise ValueError("Pas de JSON trouvé dans la réponse")
        
        except Exception as e:
            logger.warning(f"Erreur enrichissement IA: {e}")
            # Fallback simple
            return self._create_fallback_enrichment(chunk_text, document_name)
    
    def _create_fallback_enrichment(self, chunk_text: str, document_name: str) -> EnrichedContent:
        """Créer un enrichissement de fallback simple"""
        
        # Détection basique par mots-clés
        content_lower = chunk_text.lower()
        
        if any(word in content_lower for word in ['chiffre d\'affaires', 'ca ', 'ebitda', 'résultat', 'bénéfice', 'marge']):
            theme = "Analyse financière"
            commentaire = "Informations financières et indicateurs de performance"
        elif any(word in content_lower for word in ['équipe', 'direction', 'management', 'ceo', 'directeur', 'dirigeant']):
            theme = "Équipe de direction"
            commentaire = "Informations sur l'équipe dirigeante et l'organisation"
        elif any(word in content_lower for word in ['marché', 'concurrence', 'secteur', 'clients', 'segment']):
            theme = "Étude de marché"
            commentaire = "Analyse du marché et de l'environnement concurrentiel"
        elif any(word in content_lower for word in ['entreprise', 'société', 'activité', 'métier', 'business']):
            theme = "Présentation de l'entreprise"
            commentaire = "Description de l'activité et de l'organisation"
        elif any(word in content_lower for word in ['stratégie', 'objectif', 'plan', 'développement', 'croissance']):
            theme = "Stratégie et business model"
            commentaire = "Éléments de stratégie et de développement"
        elif any(word in content_lower for word in ['risque', 'opportunité', 'menace', 'défi']):
            theme = "Risques et opportunités"
            commentaire = "Analyse des risques et opportunités identifiés"
        else:
            theme = "Contenu général"
            commentaire = "Informations diverses du document"
        
        contenu_enrichi = f"{theme} : {commentaire}"
        
        logger.info(f"Fallback enrichissement: {theme}")
        
        return EnrichedContent(
            theme_principal=theme,
            commentaire_guidage=commentaire,
            contenu_enrichi=contenu_enrichi
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Générer un embedding avec cache"""
        
        # Créer clé de cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Vérifier le cache
        if text_hash in self.embedding_cache:
            self.metrics["cache_hits_embedding"] += 1
            return self.embedding_cache[text_hash]
        
        # Limiter la taille du texte
        if len(text) > 8000:
            text = text[:8000] + "..."
            logger.warning(f"Texte tronqué pour embedding: {len(text)} chars")
        
        try:
            time.sleep(0.1)  # Rate limiting
            
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            
            # Validation
            if not embedding or all(x == 0.0 for x in embedding):
                logger.error("Embedding vide généré")
                raise ValueError("Embedding vide")
            
            # Mettre en cache
            self.embedding_cache[text_hash] = embedding
            
            # Mettre à jour les métriques
            self.metrics["embeddings_generated"] += 1
            
            # Sauvegarder périodiquement
            if len(self.embedding_cache) % 20 == 0:
                self._save_caches()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erreur embedding: {e}")
            # Embedding de fallback aléatoire
            import random
            fallback_embedding = [random.uniform(-0.1, 0.1) for _ in range(768)]
            logger.warning("Utilisation d'un embedding de fallback")
            return fallback_embedding
    
    def process_chunk(self, chunk_text: str, document_name: str, 
                     folder_path: str, chunk_index: int) -> SimplifiedChunk:
        """Traiter un chunk complet avec la nouvelle approche"""
        
        # 1. Créer les métadonnées basiques
        metadata = SimplifiedMetadata(
            contenu_chunk=chunk_text,
            date_creation=time.strftime("%Y-%m-%d %H:%M:%S"),
            nom_document=document_name,
            dossier_racine=folder_path,
            chunk_index=chunk_index,
            taille_chunk=len(chunk_text)
        )
        
        # 2. Enrichir le contenu avec l'IA
        enriched_content = self.enrich_chunk_content(chunk_text, document_name)
        
        # 3. Générer l'embedding du contenu enrichi
        embedding = self.get_embedding(enriched_content.contenu_enrichi)
        
        # 4. Créer l'ID unique
        doc_hash = hashlib.md5(document_name.encode()).hexdigest()[:8]
        chunk_id = f"{doc_hash}_{chunk_index:03d}"
        
        # 5. Assembler le chunk final
        return SimplifiedChunk(
            id=chunk_id,
            metadata=metadata,
            enriched_content=enriched_content,
            embedding=embedding
        )
    
    def process_document(self, file_path: str, content: str, folder_path: str) -> List[SimplifiedChunk]:
        """Traiter un document complet avec la nouvelle approche"""
        
        document_name = os.path.basename(file_path)
        logger.info(f"📄 Traitement document: {document_name}")
        
        # Chunking intelligent selon le type de document
        chunks = self._intelligent_chunking(content, document_name)
        logger.info(f"📊 {len(chunks)} chunks créés")
        
        # Traiter chaque chunk
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            try:
                processed_chunk = self.process_chunk(
                    chunk_text=chunk_text,
                    document_name=document_name,
                    folder_path=folder_path,
                    chunk_index=i
                )
                processed_chunks.append(processed_chunk)
                
                logger.info(f"  Chunk {i}: {processed_chunk.enriched_content.theme_principal}")
                
            except Exception as e:
                logger.error(f"❌ Erreur traitement chunk {i}: {e}")
                continue
        
        # Sauvegarder les caches
        self._save_caches()
        
        logger.info(f"✅ Document traité: {len(processed_chunks)} chunks enrichis")
        return processed_chunks
    
    def _intelligent_chunking(self, text: str, document_name: str) -> List[str]:
        """Chunking intelligent selon le type de document"""
        
        # Détecter le type de document par l'extension
        file_ext = os.path.splitext(document_name.lower())[1]
        
        logger.info(f"📋 Chunking {file_ext} pour {document_name}")
        
        if file_ext == '.pdf':
            return self._chunk_pdf(text)
        elif file_ext == '.pptx':
            return self._chunk_powerpoint(text)
        elif file_ext == '.docx':
            return self._chunk_word(text)
        elif file_ext == '.xlsx':
            return self._chunk_excel(text)
        else:
            # Fallback vers chunking simple
            logger.info(f"⚠️ Type {file_ext} non reconnu, chunking simple")
            return self._simple_chunking(text)
    
    def _chunk_pdf(self, text: str) -> List[str]:
        """Chunking PDF par pages et paragraphes"""
        chunks = []
        
        # Diviser par pages d'abord
        pages = text.split('--- Page ')
        
        for page_content in pages:
            if not page_content.strip():
                continue
                
            # Garder l'info de page si présente
            if page_content.startswith('1 ---') or 'Page' in page_content[:20]:
                page_header = page_content.split('---')[0].strip() if '---' in page_content else ""
                content = page_content.split('---', 1)[-1] if '---' in page_content else page_content
            else:
                page_header = ""
                content = page_content
            
            # Diviser par paragraphes
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            current_chunk = page_header
            for paragraph in paragraphs:
                # Si ajouter ce paragraphe dépasse la taille max
                if len(current_chunk + '\n\n' + paragraph) > 2000 and current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    if current_chunk.strip():
                        current_chunk += '\n\n' + paragraph
                    else:
                        current_chunk = paragraph
            
            # Ajouter le dernier chunk de la page
            if current_chunk.strip() and len(current_chunk.strip()) > 100:
                chunks.append(current_chunk.strip())
        
        logger.info(f"📄 PDF chunking: {len(chunks)} chunks par pages/paragraphes")
        return chunks
    
    def _chunk_powerpoint(self, text: str) -> List[str]:
        """Chunking PowerPoint par slides"""
        chunks = []
        
        # Diviser par slides
        slides = text.split('--- Slide ')
        
        for slide_content in slides:
            if not slide_content.strip():
                continue
            
            # Nettoyer et garder chaque slide comme un chunk
            if slide_content.startswith('1 ---') or 'Slide' in slide_content[:20]:
                # Garder l'en-tête de slide
                full_slide = "Slide " + slide_content
            else:
                full_slide = slide_content
            
            # Limiter la taille si nécessaire
            if len(full_slide) > 3000:
                # Diviser en sous-parties si slide trop longue
                parts = full_slide.split('\n')
                current_part = ""
                for part in parts:
                    if len(current_part + '\n' + part) > 2000 and current_part.strip():
                        chunks.append(current_part.strip())
                        current_part = part
                    else:
                        current_part += '\n' + part if current_part else part
                
                if current_part.strip() and len(current_part.strip()) > 50:
                    chunks.append(current_part.strip())
            else:
                if full_slide.strip() and len(full_slide.strip()) > 50:
                    chunks.append(full_slide.strip())
        
        logger.info(f"📊 PowerPoint chunking: {len(chunks)} chunks par slides")
        return chunks
    
    def _chunk_word(self, text: str) -> List[str]:
        """Chunking Word par paragraphes et sections"""
        chunks = []
        
        # Essayer de détecter les sections (titres)
        lines = text.split('\n')
        current_section = ""
        current_content = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Détecter les titres (courts, en majuscules, ou commençant par des numéros)
            is_title = (
                len(line) < 100 and 
                (line.isupper() or 
                 line.startswith(('1.', '2.', '3.', 'I.', 'II.', 'III.', '#')) or
                 any(word in line.lower() for word in ['résumé', 'introduction', 'conclusion', 'analyse', 'stratégie']))
            )
            
            if is_title and current_content.strip():
                # Finaliser la section précédente
                section_chunk = current_section + '\n\n' + current_content if current_section else current_content
                if len(section_chunk.strip()) > 100:
                    chunks.append(section_chunk.strip())
                
                # Commencer nouvelle section
                current_section = line
                current_content = ""
            else:
                # Ajouter au contenu de la section
                if current_content:
                    current_content += '\n' + line
                else:
                    current_content = line
                
                # Si le contenu devient trop long, créer un chunk
                section_chunk = current_section + '\n\n' + current_content if current_section else current_content
                if len(section_chunk) > 2000:
                    chunks.append(section_chunk.strip())
                    current_content = ""
        
        # Ajouter le dernier chunk
        if current_content.strip():
            section_chunk = current_section + '\n\n' + current_content if current_section else current_content
            if len(section_chunk.strip()) > 100:
                chunks.append(section_chunk.strip())
        
        logger.info(f"📝 Word chunking: {len(chunks)} chunks par sections/paragraphes")
        return chunks
    
    def _chunk_excel(self, text: str) -> List[str]:
        """Chunking Excel par feuilles"""
        chunks = []
        
        # Diviser par feuilles
        sheets = text.split('--- Feuille: ')
        
        for sheet_content in sheets:
            if not sheet_content.strip():
                continue
            
            # Garder chaque feuille comme un chunk
            if sheet_content.startswith('Feuille') or '---' in sheet_content[:50]:
                full_sheet = "Feuille: " + sheet_content
            else:
                full_sheet = sheet_content
            
            # Si la feuille est très longue, la diviser en sections
            if len(full_sheet) > 4000:
                lines = full_sheet.split('\n')
                current_chunk = ""
                chunk_count = 0
                
                for line in lines:
                    if len(current_chunk + '\n' + line) > 2000 and current_chunk.strip():
                        chunk_count += 1
                        chunk_title = f"Feuille (partie {chunk_count})"
                        chunks.append(chunk_title + '\n\n' + current_chunk.strip())
                        current_chunk = line
                    else:
                        current_chunk += '\n' + line if current_chunk else line
                
                if current_chunk.strip() and len(current_chunk.strip()) > 100:
                    chunk_count += 1
                    chunk_title = f"Feuille (partie {chunk_count})"
                    chunks.append(chunk_title + '\n\n' + current_chunk.strip())
            else:
                if full_sheet.strip() and len(full_sheet.strip()) > 100:
                    chunks.append(full_sheet.strip())
        
        logger.info(f"📈 Excel chunking: {len(chunks)} chunks par feuilles")
        return chunks
    
    def _simple_chunking(self, text: str, max_size: int = 1500, overlap: int = 200) -> List[str]:
        """Chunking simple par taille avec overlap"""
        
        if not text or len(text.strip()) < 100:
            return []
        
        # Nettoyer le texte
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Essayer de diviser par paragraphes d'abord
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            paragraphs = [text]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si ajouter ce paragraphe dépasse la taille max
            if len(current_chunk + "\n\n" + paragraph) > max_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Commencer nouveau chunk avec overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    # Prendre les derniers mots pour l'overlap
                    words = current_chunk.split()
                    overlap_words = int(len(words) * overlap / len(current_chunk))
                    current_chunk = " ".join(words[-overlap_words:]) + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Ajouter au chunk actuel
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Ajouter le dernier chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filtrer les chunks trop petits ou trop gros
        filtered_chunks = []
        for chunk in chunks:
            if 100 <= len(chunk) <= 8000:  # Entre 100 chars et 8KB
                filtered_chunks.append(chunk)
            elif len(chunk) > 8000:
                # Diviser les chunks trop gros
                sub_chunks = self._force_split_large_chunk(chunk, max_size)
                filtered_chunks.extend(sub_chunks)
        
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
        
        return [chunk for chunk in sub_chunks if len(chunk.strip()) > 100]
    
    def batch_process_documents(self, file_contents: Dict[str, str]) -> List[SimplifiedChunk]:
        """Traiter plusieurs documents en batch"""
        all_chunks = []
        
        logger.info(f"🚀 Traitement de {len(file_contents)} documents...")
        
        for file_path, content in file_contents.items():
            try:
                # Extraire le dossier racine (simple)
                folder_path = str(Path(file_path).parent)
                
                chunks = self.process_document(file_path, content, folder_path)
                all_chunks.extend(chunks)
                logger.info(f"✅ {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"❌ Erreur traitement {file_path}: {e}")
                continue
        
        logger.info(f"🎉 Traitement terminé: {len(all_chunks)} chunks au total")
        return all_chunks
    
    def get_metrics(self) -> Dict[str, Any]:
        """Récupérer les métriques de performance"""
        total_enrichments = self.metrics["enrichments_generated"] + self.metrics["cache_hits_enrichment"]
        total_embeddings = self.metrics["embeddings_generated"] + self.metrics["cache_hits_embedding"]
        
        enrichment_cache_rate = (self.metrics["cache_hits_enrichment"] / total_enrichments * 100) if total_enrichments > 0 else 0
        embedding_cache_rate = (self.metrics["cache_hits_embedding"] / total_embeddings * 100) if total_embeddings > 0 else 0
        
        return {
            "enrichments_generated": self.metrics["enrichments_generated"],
            "embeddings_generated": self.metrics["embeddings_generated"],
            "enrichment_cache_hits": self.metrics["cache_hits_enrichment"],
            "embedding_cache_hits": self.metrics["cache_hits_embedding"],
            "enrichment_cache_rate": f"{enrichment_cache_rate:.1f}%",
            "embedding_cache_rate": f"{embedding_cache_rate:.1f}%",
            "total_enrichments": total_enrichments,
            "total_embeddings": total_embeddings,
            "cache_sizes": {
                "enrichment_cache": len(self.enrichment_cache),
                "embedding_cache": len(self.embedding_cache)
            }
        }
    
    def print_metrics(self):
        """Afficher les métriques de performance"""
        metrics = self.get_metrics()
        
        print("📊 Métriques de Performance:")
        print(f"   🔥 Enrichissements générés: {metrics['enrichments_generated']}")
        print(f"   🎯 Embeddings générés: {metrics['embeddings_generated']}")
        print(f"   💾 Cache enrichissement: {metrics['enrichment_cache_rate']} ({metrics['enrichment_cache_hits']}/{metrics['total_enrichments']})")
        print(f"   💾 Cache embedding: {metrics['embedding_cache_rate']} ({metrics['embedding_cache_hits']}/{metrics['total_embeddings']})")
        print(f"   📦 Taille caches: {metrics['cache_sizes']['enrichment_cache']} enrichissements, {metrics['cache_sizes']['embedding_cache']} embeddings")
    
    def clear_caches(self):
        """Vider tous les caches"""
        self.enrichment_cache = {}
        self.embedding_cache = {}
        
        # Réinitialiser les métriques
        self.metrics = {
            "embeddings_generated": 0,
            "enrichments_generated": 0,
            "cache_hits_embedding": 0,
            "cache_hits_enrichment": 0
        }
        
        # Supprimer les fichiers de cache
        for cache_file in ["enrichment_cache.json", "embedding_cache.json"]:
            cache_path = self.cache_dir / cache_file
            if cache_path.exists():
                cache_path.unlink()
        
        logger.info("🗑️ Tous les caches vidés")

# Fonction de test
def test_simplified_approach():
    """Test de la nouvelle approche simplifiée"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY manquant dans .env")
        return
    
    # Créer le gestionnaire
    manager = SimplifiedEmbeddingManager(api_key)
    
    # Test avec un exemple
    test_content = """
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
    """
    
    print("🧪 Test de l'approche simplifiée...")
    chunks = manager.process_document(
        file_path="test_business_plan.pdf",
        content=test_content,
        folder_path="EODEN/Dossiers"
    )
    
    print(f"\n✅ {len(chunks)} chunks traités:")
    for chunk in chunks:
        print(f"\n📋 CHUNK {chunk.metadata.chunk_index}:")
        print(f"   📊 Thème: {chunk.enriched_content.theme_principal}")
        print(f"   💬 Guide: {chunk.enriched_content.commentaire_guidage}")
        print(f"   🎯 Vectorisé: {chunk.enriched_content.contenu_enrichi}")
        print(f"   📄 Source: {chunk.metadata.nom_document}")
        print(f"   📁 Dossier: {chunk.metadata.dossier_racine}")
        print(f"   📏 Taille: {chunk.metadata.taille_chunk} chars")
        print(f"   🔗 Embedding: {len(chunk.embedding)} dimensions")

if __name__ == "__main__":
    test_simplified_approach()