# Configuration OCR pour EODEN POC

## 🔍 Fonctionnalités OCR

Le système peut maintenant extraire le texte des **images** dans les documents avec **3 moteurs OCR** :

- ✅ **PDF** : Images scannées, graphiques avec texte
- ✅ **PowerPoint** : Images dans les slides  
- ✅ **Word** : Images intégrées (prochaine version)
- ✅ **Excel** : Graphiques avec texte (prochaine version)

## 🥇 **Gemini 2.5 Flash OCR** (Recommandé)

### Avantages révolutionnaires
- 🚀 **3x plus rapide** que Gemini 1.5 avec performances supérieures
- 💰 **75% moins cher** - Optimisation budgétaire maximale
- 🧠 **Compréhension contextuelle révolutionnaire** des documents d'investissement
- 📊 **Extraction multi-niveau** : OCR + Analyse + Insights business + Prédictions
- 🔗 **API unifiée** - Même clé que les embeddings
- 🎯 **Prompts complexes** optimisés pour l'analyse financière
- 📈 **Support 8K+ tokens** - Analyses ultra-approfondies

### Résultat avec Gemini 2.5 Flash
```
[Image 1 - Gemini 2.5 Analysis]
TEXTE EXTRAIT:
Évolution du Chiffre d'Affaires
Q4 2021: 1.2M€ | Q4 2022: 1.8M€ | Q4 2023: 2.1M€
Objectif 2024: 2.8M€

MÉTRIQUES FINANCIÈRES:
- CA Q4 2021: 1 200 000€
- CA Q4 2022: 1 800 000€ (croissance +50%)
- CA Q4 2023: 2 100 000€ (croissance +17%)
- Projection 2024: 2 800 000€ (croissance +33%)

DONNÉES TEMPORELLES:
- TCAM 2021-2023: +32% (très forte croissance)
- Ralentissement 2023: -33pts vs 2022 (attention)
- Rebond prévu 2024: +33% (ambitieux)

INSIGHTS CLÉS:
- Croissance robuste mais cycle ralentit
- Objectif 2024 nécessite accélération
- Tendance générale positive pour investisseurs
- Surveillance nécessaire sur Q1-Q2 2024

TYPE DE CONTENU:
Graphique en barres temporelles avec projections
```

## 🛠️ Installation Tesseract (Gratuit)

### Windows
```bash
# 1. Télécharger Tesseract depuis GitHub
https://github.com/UB-Mannheim/tesseract/wiki

# 2. Installer avec support français
# Cocher "Additional language data (French)" lors de l'installation

# 3. Ajouter au PATH système
C:\Program Files\Tesseract-OCR

# 4. Installer Python packages
pip install pytesseract Pillow opencv-python
```

### Linux/Mac
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-fra

# macOS avec Homebrew
brew install tesseract tesseract-lang

# Python packages
pip install pytesseract Pillow opencv-python
```

## 🔧 Configuration

### Dans `config/settings.yaml` :

```yaml
ocr:
  enabled: true
  preferred_engine: "tesseract"
  
  tesseract:
    languages: ["fra", "eng"]
    config: "--psm 6"
  
  quality:
    min_confidence: 0.5
    max_image_size_mb: 10
```

### Variables d'environnement (si nécessaire) :

```bash
# Si Tesseract n'est pas dans le PATH
export TESSERACT_CMD="/usr/local/bin/tesseract"
```

## 🚀 Google Vision API (Optionnel, Plus Précis)

### 1. Créer un projet Google Cloud
1. Aller sur [Google Cloud Console](https://console.cloud.google.com)
2. Créer un nouveau projet
3. Activer l'API Vision

### 2. Créer une clé de service
```bash
# Télécharger le fichier JSON des credentials
# Le placer dans config/google-vision-credentials.json
```

### 3. Configuration
```yaml
ocr:
  preferred_engine: "google_vision"
  google_vision:
    credentials_file: "config/google-vision-credentials.json"
```

### 4. Variables d'environnement
```bash
export GOOGLE_APPLICATION_CREDENTIALS="config/google-vision-credentials.json"
```

## 📊 Comment ça marche

### Déclenchement automatique
L'OCR se déclenche automatiquement quand :
- Une page PDF a moins de 100 caractères de texte
- Une slide PowerPoint a moins de 50 caractères
- Le document semble être principalement constitué d'images

### Stratégie en cascade (optimisée avec Gemini 2.5)
1. **Gemini 2.5 Flash** (révolutionnaire, 3x plus rapide, 75% moins cher) en premier
2. **Tesseract** (gratuit, local) en fallback
3. **Google Vision** (payant, précis) en dernier recours
4. **Ignorer** si les trois échouent

### Résultat dans les chunks
Le texte OCR est ajouté selon le moteur utilisé :

```
Contenu normal du document...

[Image 1 - Gemini 2.5 Analysis]
CONTENU TEXTUEL:
Résultats financiers Q3 2023
Chiffre d'affaires : 2.1M€ | EBITDA : 18% | Marge nette : 12%

DONNÉES BUSINESS:
- CA Q3 2023: 2 100 000€ (+15% vs Q3 2022)
- EBITDA: 378 000€ (marge 18%)
- Résultat net: 252 000€ (marge 12%)
- Croissance annualisée: +22%

ÉLÉMENTS VISUELS:
Tableau de bord financier avec codes couleur vert (croissance positive)
et graphiques barres comparatives vs années précédentes

MESSAGE STRATÉGIQUE:
Démonstration de la solidité financière et de la trajectoire de croissance
pour rassurer les investisseurs sur la performance opérationnelle

[Image 2] Graphique evolution CA (Tesseract fallback)
2021: 1.2M€
2022: 1.8M€  
2023: 2.1M€
```

## 🎯 Avantages

### Documents scannés
- ✅ PDF scannés entièrement récupérés
- ✅ Anciennes présentations numérisées
- ✅ Rapports avec graphiques

### Graphiques et tableaux
- ✅ Extraction de données chiffrées
- ✅ Légendes et annotations
- ✅ Tableaux complexes dans les images

### Multilingue
- ✅ Français + Anglais simultanément
- ✅ Reconnaissance adaptée au contexte business

## 🔧 Dépannage

### Erreur "tesseract not found"
```bash
# Vérifier l'installation
tesseract --version

# Configurer le chemin si nécessaire
export PATH=$PATH:/usr/local/bin
```

### Performance lente
```yaml
ocr:
  quality:
    max_image_size_mb: 5  # Réduire la taille max
    skip_small_images: true  # Ignorer petites images
```

### Précision faible
```yaml
ocr:
  tesseract:
    config: "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz€%.,:-"
```

## 💡 Conseils

### Optimiser la qualité
- Utilisez des documents en haute résolution
- Google Vision est plus précis pour les graphiques complexes
- Tesseract fonctionne bien pour du texte simple

### Coût vs Qualité
- **Tesseract** : Gratuit, bon pour 80% des cas
- **Google Vision** : ~1.5€/1000 images, excellent pour tout

### Debug OCR
```yaml
debug:
  save_ocr_results: true  # Voir les résultats dans logs/
```

Les résultats OCR seront visibles dans les logs pour vérifier la qualité !