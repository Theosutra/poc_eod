# POC EODEN - RAG Pipeline

Syst�me de g�n�ration automatique de contenu bas� sur l'analyse de documents d'investissement.

## <� Fonctionnalit�s

- **Extraction automatique** de documents depuis Google Drive
- **Vectorisation intelligente** avec chunking adaptatif par type de document
- **Recherche s�mantique** dans une base vectorielle (Pinecone/Elasticsearch)
- **G�n�ration de contenu** via Gemini AI

## =� Structure

```
poc_eoden/
   src/                          # Code source principal
      main.py                   # Pipeline RAG principal
      embedding_manager.py      # Gestion des embeddings et chunking
      content_generator.py      # G�n�ration de contenu IA
      vector_store.py           # Interface base vectorielle
      drive_connector.py        # Connexion Google Drive
      document_processor.py     # Traitement des documents
      template_manager.py       # Gestion des templates
   config/
      settings.yaml             # Configuration principale
      credentials.json          # Cl�s Google Drive
   data/
      cache/                    # Documents t�l�charg�s
      embeddings/               # Cache des embeddings
      output/                   # Fichiers g�n�r�s
   test/                         # Tests unitaires
   logs/                         # Fichiers de log
   requirements.txt              # D�pendances Python
   test_generator.py             # Test du g�n�rateur
```

## =� Installation

1. **Cloner et naviguer** :
```bash
cd poc_eoden
```

2. **Activer l'environnement virtuel** :
```bash
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Installer les d�pendances** :
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement** :
```bash
export GEMINI_API_KEY=votre_cl�_gemini
export PINECONE_API_KEY=votre_cl�_pinecone  # optionnel
```

## >� Utilisation

### Ex�cution compl�te du pipeline :
```bash
python src/main.py
```

### Test du g�n�rateur de contenu :
```bash
python test_generator.py
```

### Test des composants individuels :
```bash
# Test Google Drive
python test/test_drive.py

# Test direct (sans Drive)
python test/test_direct.py
```

## =� Status du Syst�me

| Composant | Status | Description |
|-----------|--------|-------------|
| **Cache Embeddings** |  Fonctionnel | 19 embeddings (768 dims) |
| **Documents Source** |  OK | 4 PDFs (7.8MB) |
| **Chunking Intelligent** |  OK | Strat�gies par type |
| **Base Vectorielle** | � Configuration | Pinecone configur� |
| **G�n�ration IA** | � D�pendances | Gemini API requis |

## =' Configuration

### Fichier `config/settings.yaml` :
```yaml
vector_store:
  type: "pinecone"
  index_name: "eoden-investment-docs"
  dimension: 768
  config:
    api_key: ${PINECONE_API_KEY}
    environment: "gcp-starter"

google_drive:
  credentials_file: "config/credentials.json"
  source_folders:
    - name: "Documents Investissement"
      recursive: true
```

## =� M�triques

- **Documents trait�s** : 4 PDFs
- **Chunks g�n�r�s** : 19 chunks
- **Qualit� embeddings** : Distribution normale
- **Couverture** : 4.8 chunks/document

## � Pr�requis

1. **Python 3.8+** avec environnement virtuel
2. **Cl� API Gemini** (obligatoire)
3. **Cl� API Pinecone** (optionnel, pour base vectorielle cloud)
4. **Credentials Google Drive** (pour synchro automatique)

## = Troubleshooting

### Probl�me : ImportError modules manquants
**Solution** : V�rifier que le venv est activ� et d�pendances install�es

### Probl�me : G�n�ration vide
**Solution** : V�rifier GEMINI_API_KEY et connectivit� r�seau

### Probl�me : Aucun document trouv�
**Solution** : V�rifier cache dans `data/cache/` et permissions Google Drive

## =� Logs

Les logs sont disponibles dans `logs/poc.log` pour le debugging.