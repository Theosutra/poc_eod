"""
Générateur de contenu pour les notes d'investissement
Utilise la base vectorielle pour créer du contenu intelligent
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv

# Imports locaux
from vector_store import VectorManager
from embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

@dataclass
class GeneratedSection:
    """Section générée avec métadonnées"""
    title: str
    content: str
    confidence_score: float
    sources_used: List[str]
    chunk_count: int

class ContentGenerator:
    """Générateur de contenu intelligent"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        load_dotenv()
        
        # Initialiser les composants
        self.embedding_manager = EmbeddingManager(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.vector_manager = VectorManager(config_path)
        
        # Modèle de génération
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        
        logger.info("✅ Content Generator initialisé")
    
    def search_relevant_chunks(self, query: str, top_k: int = 15, 
                             filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Rechercher les chunks pertinents pour une requête - VERSION CORRIGÉE"""
        
        try:
            # Générer l'embedding de la requête
            query_embedding = self.embedding_manager.get_embedding(query)
            
            # CORRECTION: Vérifier que l'embedding est valide
            if not query_embedding or all(x == 0.0 for x in query_embedding):
                logger.error(f"Embedding invalide pour '{query}'")
                return []
            
            # Rechercher dans la base vectorielle avec plus de résultats
            results = self.vector_manager.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            logger.info(f"🔍 Recherche '{query}': {len(results)} chunks trouvés")
            
            # CORRECTION: Vérifier le type des résultats
            if not results:
                logger.warning(f"Aucun résultat pour '{query}'")
                return []
            
            # Convertir en format utilisable
            relevant_chunks = []
            for result in results:
                try:
                    # CORRECTION: Gestion plus robuste des métadonnées
                    metadata = result.metadata if hasattr(result, 'metadata') and result.metadata else {}
                    
                    chunk_data = {
                        "content": result.content if hasattr(result, 'content') else str(result),
                        "score": float(result.score) if hasattr(result, 'score') else 0.0,
                        "source": metadata.get("source_file", "Unknown"),
                        "section": metadata.get("section_title", "Unknown"),
                        "themes": metadata.get("themes", []),
                        "business_context": metadata.get("business_context", ""),
                        "confidence": metadata.get("confidence_score", 0.0)
                    }
                    
                    # Filtrer les chunks vides
                    if chunk_data["content"] and len(chunk_data["content"].strip()) > 20:
                        relevant_chunks.append(chunk_data)
                        
                except Exception as e:
                    logger.warning(f"Erreur traitement résultat: {e}")
                    continue
            
            # Debug: afficher les meilleurs résultats
            logger.info(f"📊 Top 3 résultats pour '{query}':")
            for i, chunk in enumerate(relevant_chunks[:3], 1):
                source_name = os.path.basename(chunk['source']) if chunk['source'] != "Unknown" else "Unknown"
                logger.info(f"  {i}. Score: {chunk['score']:.3f} - Source: {source_name}")
                logger.info(f"     Contenu: {chunk['content'][:100]}...")
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche '{query}': {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_section_content(self, section_title: str, search_queries: List[str],
                               max_chunks: int = 8) -> GeneratedSection:
        """Générer le contenu d'une section spécifique"""
        
        logger.info(f"📝 Génération de la section: {section_title}")
        
        # Collecter les chunks pertinents pour toutes les requêtes
        all_chunks = []
        for query in search_queries:
            chunks = self.search_relevant_chunks(query, top_k=max_chunks//len(search_queries) + 2)
            all_chunks.extend(chunks)
        
        # Déduplication par source et score
        seen_content = set()
        unique_chunks = []
        for chunk in all_chunks:
            content_hash = hash(chunk["content"][:100])  # Hash des 100 premiers chars
            if content_hash not in seen_content:
                unique_chunks.append(chunk)
                seen_content.add(content_hash)
        
        # Trier par score de pertinence
        unique_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Prendre les meilleurs chunks
        best_chunks = unique_chunks[:max_chunks]
        
        if not best_chunks:
            logger.warning(f"❌ Aucun chunk trouvé pour {section_title}")
            return GeneratedSection(
                title=section_title,
                content="Aucune information disponible dans les documents fournis.",
                confidence_score=0.0,
                sources_used=[],
                chunk_count=0
            )
        
        # Construire le prompt de génération - STYLE PROFESSIONNEL DIRECT
        context_text = self._build_context_from_chunks(best_chunks)
        
        prompt = f"""
        Vous êtes un analyste financier senior rédigeant une note d'investissement pour un comité d'investissement.
        
        SECTION: {section_title}
        
        INFORMATIONS DISPONIBLES:
        {context_text}
        
        INSTRUCTIONS STRICTES:
        1. Rédigez un texte professionnel DIRECT sans jamais mentionner les sources ("le document indique", "selon le rapport", etc.)
        2. Présentez les faits comme des affirmations directes et factuelles
        3. Style: Note d'investissement institutionnelle (comme Goldman Sachs ou McKinsey)
        4. Longueur: 400-800 mots, dense et informatif
        5. Structure avec des sous-titres si pertinent
        6. Chiffres précis et données factuelles en priorité
        7. Évitez les formulations vagues ou conditionnelles
        
        EXEMPLE DE STYLE:
        ❌ "Le document mentionne que l'entreprise réalise..."
        ✅ "L'entreprise réalise un chiffre d'affaires de X€..."
        
        ❌ "Selon les informations disponibles..."
        ✅ "L'activité principale porte sur..."
        
        RÉDIGEZ LA SECTION:
        """
        
        try:
            # Générer le contenu
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=1500,
                    top_p=0.8
                )
            )
            
            generated_content = response.text.strip()
            
            # Calculer le score de confiance
            avg_score = sum(chunk["score"] for chunk in best_chunks) / len(best_chunks)
            confidence_score = min(avg_score * 1.2, 1.0)  # Boost léger
            
            # Sources utilisées
            sources_used = list(set(chunk["source"] for chunk in best_chunks))
            
            logger.info(f"✅ Section '{section_title}' générée: {len(generated_content)} chars")
            
            return GeneratedSection(
                title=section_title,
                content=generated_content,
                confidence_score=confidence_score,
                sources_used=sources_used,
                chunk_count=len(best_chunks)
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur génération section {section_title}: {e}")
            return GeneratedSection(
                title=section_title,
                content=f"Erreur lors de la génération: {str(e)}",
                confidence_score=0.0,
                sources_used=[],
                chunk_count=0
            )
    
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Construire le contexte à partir des chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source_name = os.path.basename(chunk["source"]).replace(".pdf", "")
            
            context_part = f"""
            === DOCUMENT {i}: {source_name} ===
            Section: {chunk["section"]}
            Thèmes: {', '.join(chunk["themes"])}
            Score: {chunk["score"]:.3f}
            
            Contenu:
            {chunk["content"]}
            
            """
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_company_presentation_only(self) -> GeneratedSection:
        """Générer UNIQUEMENT une présentation d'entreprise complète - VERSION HYBRIDE"""
        
        logger.info("🏢 Génération de la présentation d'entreprise avec recherche hybride")
        
        # STRATÉGIE 1: Recherche ciblée par sections
        section_queries = [
            "Présentation de l'Entreprise",
            "Modèle d'affaires", 
            "business model",
            "activité principale"
        ]
        
        # STRATÉGIE 2: Recherche par noms d'entreprise identifiés
        company_queries = [
            "R-Group", "R Group", "RGroup",
            "InnoTech", "InnoTech Solutions",
            "Dynamia Invest", 
            "Reno Energy", "Reno",
            "Pony Energy"
        ]
        
        # STRATÉGIE 3: Recherche par contexte business
        business_queries = [
            "entreprise société",
            "activité métier",
            "secteur d'activité",
            "organisation structure"
        ]
        
        # STRATÉGIE 4: Recherche par thèmes identifiés
        theme_queries = [
            "Solutions logicielles",
            "Énergie renouvelable", 
            "Rénovation énergétique",
            "Transition énergétique"
        ]
        
        all_queries = section_queries + company_queries + business_queries + theme_queries
        
        # Collecter TOUS les chunks avec scores
        all_chunks_with_score = []
        
        for query in all_queries:
            try:
                chunks = self.search_relevant_chunks(query, top_k=15)
                for chunk in chunks:
                    if chunk["score"] > 0.5:  # Seuil de pertinence
                        chunk_enhanced = chunk.copy()
                        chunk_enhanced["search_query"] = query
                        all_chunks_with_score.append(chunk_enhanced)
            except Exception as e:
                logger.warning(f"Erreur recherche '{query}': {e}")
                continue
        
        logger.info(f"📊 Total chunks collectés: {len(all_chunks_with_score)}")
        
        # DÉDUPLICATION INTELLIGENTE - moins agressive
        unique_chunks = []
        seen_sources_content = {}
        
        # Grouper par source et garder les meilleurs par source
        for chunk in all_chunks_with_score:
            source = chunk["source"]
            content_start = chunk["content"][:50]  # Plus court pour moins filtrer
            
            key = f"{source}_{content_start}"
            
            if key not in seen_sources_content:
                seen_sources_content[key] = chunk
            else:
                # Garder le chunk avec le meilleur score
                if chunk["score"] > seen_sources_content[key]["score"]:
                    seen_sources_content[key] = chunk
        
        unique_chunks = list(seen_sources_content.values())
        
        # Trier par score ET privilégier certaines sections
        def chunk_priority(chunk):
            score = chunk["score"]
            
            # Bonus pour les sections importantes
            section = chunk.get("section", "").lower()
            if "présentation" in section and "entreprise" in section:
                score += 0.2
            elif "modèle" in section or "business" in section:
                score += 0.15
            elif "activité" in section:
                score += 0.1
            
            # Bonus pour les entreprises principales
            content = chunk["content"].lower()
            if "r-group" in content or "innotech" in content:
                score += 0.1
            
            return score
        
        unique_chunks.sort(key=chunk_priority, reverse=True)
        
        # Prendre plus de chunks pour avoir plus de contexte
        best_chunks = unique_chunks[:15]  
        
        logger.info(f"📊 Après déduplication: {len(unique_chunks)} chunks uniques, {len(best_chunks)} sélectionnés")
        
        if not best_chunks:
            logger.warning(f"❌ Aucun chunk trouvé pour la présentation")
            return GeneratedSection(
                title="Présentation de l'Entreprise",
                content="Aucune information détaillée disponible dans les documents fournis.",
                confidence_score=0.0,
                sources_used=[],
                chunk_count=0
            )
        
        # Log des chunks sélectionnés pour debug
        logger.info(f"🎯 Chunks sélectionnés:")
        for i, chunk in enumerate(best_chunks[:5], 1):
            source_name = os.path.basename(chunk["source"])
            logger.info(f"  {i}. Score: {chunk['score']:.3f} - {source_name}")
            logger.info(f"     Section: {chunk.get('section', 'Unknown')}")
            logger.info(f"     Requête: {chunk.get('search_query', 'N/A')}")
            logger.info(f"     Contenu: {chunk['content'][:100]}...")
        
        # Construire le contexte enrichi
        context_text = self._build_company_context(best_chunks)
        
        prompt = f"""
        Vous êtes un analyste financier senior rédigeant la section "Présentation de l'Entreprise" d'une note d'investissement institutionnelle.
        
        Vous avez collecté des informations sur plusieurs entreprises du secteur de l'énergie et de la technologie.
        
        INFORMATIONS COLLECTÉES:
        {context_text}
        
        INSTRUCTIONS:
        1. Identifiez les ENTREPRISES PRINCIPALES mentionnées (R-Group, InnoTech, Dynamia Invest, etc.)
        2. Rédigez une présentation structurée couvrant:
           - Vue d'ensemble des activités
           - Structure du groupe/des entreprises
           - Secteurs d'intervention
           - Modèles économiques
           - Positionnement concurrentiel
        
        3. Style: Note d'investissement professionnelle
        4. Ton: Factuel, sans mentionner les sources
        5. Longueur: 600-1000 mots
        6. Structurez avec des sous-titres
        
        RÉDIGEZ LA PRÉSENTATION DES ENTREPRISES:
        """
        
        try:
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2500,
                    top_p=0.9
                )
            )
            
            generated_content = response.text.strip()
            
            # Score de confiance basé sur la qualité des chunks
            weighted_score = sum(chunk["score"] * chunk.get("confidence", 0.5) for chunk in best_chunks)
            avg_score = weighted_score / len(best_chunks) if best_chunks else 0
            confidence_score = min(avg_score * 1.1, 1.0)
            
            sources_used = list(set(chunk["source"] for chunk in best_chunks))
            
            logger.info(f"✅ Présentation générée: {len(generated_content)} chars, confiance {confidence_score:.2f}")
            
            return GeneratedSection(
                title="Présentation de l'Entreprise",
                content=generated_content,
                confidence_score=confidence_score,
                sources_used=sources_used,
                chunk_count=len(best_chunks)
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur génération présentation: {e}")
            return GeneratedSection(
                title="Présentation de l'Entreprise",
                content=f"Erreur lors de la génération: {str(e)}",
                confidence_score=0.0,
                sources_used=[],
                chunk_count=0
            )
    
    def _build_company_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Construire un contexte spécialisé pour présentation d'entreprise"""
        context_parts = []
        
        # Grouper par source pour une meilleure organisation
        sources_chunks = {}
        for chunk in chunks:
            source = chunk["source"]
            if source not in sources_chunks:
                sources_chunks[source] = []
            sources_chunks[source].append(chunk)
        
        for source, source_chunks in sources_chunks.items():
            source_name = os.path.basename(source).replace(".pdf", "")
            context_parts.append(f"\n=== DOCUMENT: {source_name} ===")
            
            for chunk in source_chunks:
                section = chunk.get("section", "Section inconnue")
                themes = ", ".join(chunk.get("themes", []))
                
                context_part = f"""
SECTION: {section}
THÈMES: {themes}
SCORE: {chunk["score"]:.3f}

{chunk["content"]}
"""
                context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_rich_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Construire un contexte riche à partir des chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source_name = os.path.basename(chunk["source"]).replace(".pdf", "")
            
            context_part = f"""
            INFORMATION {i} [Score: {chunk["score"]:.3f}]:
            {chunk["content"]}
            """
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def export_to_text(self, sections: Dict[str, GeneratedSection], 
                      output_path: str = "data/output/presentation_entreprise.txt") -> str:
        """Exporter la présentation en fichier texte"""
        
        # Créer le dossier de sortie
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Construire le contenu complet
        content_lines = []
        
        # En-tête
        content_lines.extend([
            "=" * 80,
            "PRÉSENTATION D'ENTREPRISE",
            "Généré automatiquement par EODEN POC",
            f"Date: {self._get_current_date()}",
            "=" * 80,
            ""
        ])
        
        # Statistiques globales
        total_sources = set()
        total_chunks = 0
        avg_confidence = 0
        
        for section in sections.values():
            total_sources.update(section.sources_used)
            total_chunks += section.chunk_count
            avg_confidence += section.confidence_score
        
        avg_confidence = avg_confidence / len(sections) if sections else 0
        
        content_lines.extend([
            "📊 STATISTIQUES DE GÉNÉRATION",
            f"• Sections générées: {len(sections)}",
            f"• Documents sources: {len(total_sources)}",
            f"• Chunks utilisés: {total_chunks}",
            f"• Confiance moyenne: {avg_confidence:.2f}/1.0",
            "",
            "📁 SOURCES UTILISÉES:",
        ])
        
        for i, source in enumerate(sorted(total_sources), 1):
            source_name = os.path.basename(source)
            content_lines.append(f"  {i}. {source_name}")
        
        content_lines.extend(["", "=" * 80, ""])
        
        # Sections générées
        for section_title, section in sections.items():
            content_lines.extend([
                f"## {section_title.upper()}",
                f"Confiance: {section.confidence_score:.2f}/1.0 | Chunks: {section.chunk_count} | Sources: {len(section.sources_used)}",
                "-" * 60,
                "",
                section.content,
                "",
                "=" * 80,
                ""
            ])
        
        # Pied de page
        content_lines.extend([
            "NOTES:",
            "• Cette présentation a été générée automatiquement à partir des documents fournis",
            "• Les scores de confiance indiquent la pertinence des informations trouvées",
            "• Une relecture et validation manuelle est recommandée",
            ""
        ])
        
        # Écrire le fichier
        full_content = "\n".join(content_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        
        logger.info(f"✅ Présentation exportée: {output_path}")
        return output_path
    
    def _get_current_date(self) -> str:
        """Obtenir la date actuelle formatée"""
        import datetime
        return datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

# Fonction principale pour tester
def main():
    """Fonction principale de test"""
    print("🚀 Test du générateur de contenu")
    
    try:
        # Créer le générateur
        generator = ContentGenerator()
        
        # Générer la présentation
        print("📝 Génération de la présentation d'entreprise...")
        sections = generator.generate_company_presentation()
        
        # Exporter en fichier texte
        output_path = generator.export_to_text(sections)
        
        print(f"✅ Présentation générée et sauvegardée: {output_path}")
        
        # Afficher un résumé
        print(f"\n📊 Résumé:")
        for title, section in sections.items():
            print(f"  • {title}: {len(section.content)} chars, confiance {section.confidence_score:.2f}")
        
        print(f"\n📖 Ouvrir le fichier pour voir le résultat complet !")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()