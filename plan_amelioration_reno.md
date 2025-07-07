# Plan d'Amélioration - Problème Reno Energy dans le RAG

## Diagnostic

**Problème identifié** : Le système RAG ne récupère pas correctement les informations sur Reno Energy car l'extraction PDF ne fonctionne que sur les métadonnées et non sur le contenu textuel principal.

**Preuve** :
- ✅ Reno Energy est présent dans les métadonnées PDF (`<</Title(Reno.Energy)`, `Slide 27: Projet structurant Reno-As-A-Service`)
- ❌ Reno Energy n'est pas dans le contenu textuel extrait
- ⚠️ Les sorties générées mentionnent "rénovation énergétique" mais pas "Reno Energy" spécifiquement

## Causes Racines

1. **CRITIQUE** : Extraction PDF défaillante
   - Le code utilise `pymupdf` mais la bibliothèque n'est pas installée
   - Le fallback sur le contenu brut ne fonctionne pas correctement

2. **MAJEUR** : Chunking sur contenu incomplet
   - Les chunks sont créés à partir du contenu mal extrait
   - Les métadonnées importantes (titres de slides) ne sont pas préservées

3. **POTENTIEL** : Requêtes de recherche insuffisantes
   - Les variantes de "Reno Energy" ne sont pas toutes testées

## Plan d'Action

### 🔴 PRIORITÉ 1 : Corriger l'extraction PDF

#### Actions immédiates :
1. **Installer PyMuPDF** dans l'environnement
   ```bash
   pip install pymupdf
   ```

2. **Vérifier l'installation dans le code**
   - Le code `main.py` ligne 214 utilise `import pymupdf` 
   - Cela devrait fonctionner une fois la bibliothèque installée

3. **Ajouter des fallbacks d'extraction**
   - Si PyMuPDF échoue, essayer pdfplumber ou PyPDF2
   - Ajouter une extraction des métadonnées (table des matières, titres)

#### Code à modifier dans `main.py` :

```python
def _extract_pdf_content(self, file_path: Path) -> str:
    """Extraire le contenu d'un PDF avec multiples méthodes"""
    text = ""
    
    # Méthode 1: PyMuPDF (préférée)
    try:
        import pymupdf as fitz
        doc = fitz.open(str(file_path))
        
        # Extraire la table des matières
        toc = doc.get_toc()
        if toc:
            text += "\n=== TABLE DES MATIÈRES ===\n"
            for level, title, page in toc:
                text += f"{'  ' * level}{title} (Page {page})\n"
            text += "\n"
        
        # Extraire le contenu des pages
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()
        
        doc.close()
        logger.info(f"✅ Extraction PyMuPDF réussie: {len(text)} caractères")
        return text.strip()
        
    except ImportError:
        logger.warning("PyMuPDF non disponible, essai méthode alternative")
    except Exception as e:
        logger.warning(f"Erreur PyMuPDF: {e}, essai méthode alternative")
    
    # Méthode 2: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num} ---\n"
                    text += page_text
        
        logger.info(f"✅ Extraction pdfplumber réussie: {len(text)} caractères")
        return text.strip()
        
    except ImportError:
        logger.warning("pdfplumber non disponible")
    except Exception as e:
        logger.warning(f"Erreur pdfplumber: {e}")
    
    # Méthode 3: PyPDF2
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num} ---\n"
                text += page_text
        
        logger.info(f"✅ Extraction PyPDF2 réussie: {len(text)} caractères")
        return text.strip()
        
    except ImportError:
        logger.warning("PyPDF2 non disponible")
    except Exception as e:
        logger.warning(f"Erreur PyPDF2: {e}")
    
    # Si tout échoue
    logger.error(f"❌ Impossible d'extraire le contenu de {file_path}")
    return ""
```

### 🟡 PRIORITÉ 2 : Améliorer le chunking

#### Actions :
1. **Préserver les métadonnées importantes** dans `embedding_manager.py`
2. **Détecter les entités** comme "Reno Energy" lors du chunking
3. **Éviter de fragmenter** les informations sur les entreprises

#### Code à ajouter dans `embedding_manager.py` :

```python
def detect_entities(self, text: str) -> List[str]:
    """Détecter les entités importantes dans le texte"""
    entities = []
    
    # Entreprises connues
    companies = [
        "Reno Energy", "Reno.Energy", "R-Group", "InnoTech", 
        "Dynamia Invest", "Pony Energy"
    ]
    
    text_lower = text.lower()
    for company in companies:
        if company.lower() in text_lower:
            entities.append(company)
    
    return entities

def intelligent_chunking(self, text: str, doc_type: DocumentType, 
                       source_file: str) -> List[str]:
    """Chunking intelligent avec préservation des entités"""
    
    # Détecter les entités importantes
    important_entities = self.detect_entities(text)
    
    if important_entities:
        logger.info(f"Entités détectées: {important_entities}")
    
    # Appliquer le chunking existant
    chunks = self._hierarchical_split(text, ...)
    
    # Post-traitement : vérifier que les entités ne sont pas fragmentées
    improved_chunks = []
    for chunk in chunks:
        # Si un chunk contient une entité partielle, essayer de le fusionner
        for entity in important_entities:
            if any(word in chunk.lower() for word in entity.lower().split()):
                # Vérifier que l'entité complète est présente
                if entity.lower() not in chunk.lower():
                    logger.warning(f"Entité {entity} possiblement fragmentée dans chunk")
        
        improved_chunks.append(chunk)
    
    return improved_chunks
```

### 🟢 PRIORITÉ 3 : Enrichir les requêtes de recherche

#### Actions :
1. **Ajouter plus de variantes** de "Reno Energy" dans `content_generator.py`
2. **Tester différentes formulations** de requêtes

#### Code à modifier dans `content_generator.py` :

```python
def generate_company_presentation_only(self) -> GeneratedSection:
    """Générer UNIQUEMENT une présentation d'entreprise complète - VERSION AMÉLIORÉE"""
    
    # STRATÉGIE ENRICHIE pour Reno Energy
    reno_queries = [
        # Variations directes
        "Reno Energy", "Reno.Energy", "RenoEnergy",
        "Reno", "Energy",
        
        # Contexte business
        "Reno entreprise", "Reno société", "Reno activité",
        "Energy entreprise", "Energy société",
        
        # Services
        "Reno-As-A-Service", "Reno as a Service", "RaaS",
        "service Reno", "solution Reno",
        
        # Secteur
        "rénovation énergétique Reno", "transition énergétique Reno",
        "efficacité énergétique Reno", "performance énergétique Reno",
        
        # Projets
        "projet Reno", "projet structurant Reno",
        "développement Reno", "innovation Reno"
    ]
```

### 🔵 PRIORITÉ 4 : Vérification et tests

#### Actions :
1. **Script de test complet** pour vérifier l'extraction
2. **Tests de régression** pour s'assurer que Reno Energy est bien récupéré

## Indicateurs de Succès

- [ ] **Extraction PDF** : Le contenu textuel est correctement extrait (pas seulement les métadonnées)
- [ ] **Chunking** : Les chunks contiennent des informations complètes sur Reno Energy
- [ ] **Recherche** : Les requêtes "Reno Energy" retournent des résultats pertinents avec un score > 0.7
- [ ] **Génération** : La présentation d'entreprise mentionne spécifiquement Reno Energy avec des détails

## Timeline

- **J+1** : Installer PyMuPDF et tester l'extraction
- **J+2** : Modifier le code d'extraction avec fallbacks
- **J+3** : Améliorer le chunking et les requêtes
- **J+4** : Tests complets et validation
- **J+5** : Déploiement et monitoring

## Risques et Mitigation

- **Risque** : PyMuPDF ne fonctionne toujours pas
  - **Mitigation** : Utiliser pdfplumber ou extraction manuelle
  
- **Risque** : Le document ne contient réellement pas d'informations sur Reno Energy
  - **Mitigation** : Vérification manuelle du PDF source dans Google Drive

- **Risque** : Les informations sont trop fragmentées dans le PDF
  - **Mitigation** : Améliorer l'extraction pour préserver la structure