# 💎 Diamonds ML — Prédiction du Prix d'un Diamant

Application de Machine Learning complète pour prédire le prix d'un diamant à partir de ses caractéristiques physiques et de qualité.

---

## 🚀 Demo

> 🔗 [Lancer l'application](https://cesarlome4-diamonds-ml.streamlit.app) *(après déploiement sur Streamlit Cloud)*

---

## 📋 Description du Projet

Ce projet couvre un pipeline de Machine Learning supervisé complet, de l'exploration des données jusqu'au déploiement d'une application interactive.

**Dataset :** Diamonds (seaborn) — 53 940 diamants, 9 features, cible : `price`

| Étape | Détail |
|---|---|
| 📊 EDA | Exploration, corrélations, distribution |
| 🧹 Nettoyage | Suppression outliers (IQR×3), valeurs aberrantes |
| ⚙️ Feature Engineering | `volume = x×y×z`, `ratio_depth_table` |
| 🔄 Encodage | `OrdinalEncoder` (cut, color, clarity) |
| 📏 Normalisation | `MinMaxScaler` via `ColumnTransformer` |
| 🎯 Sélection | `SelectKBest(f_regression, k=10)` |
| 🤖 Modèles | 8 modèles (régression + classification) |
| 🔧 Optimisation | `GridSearchCV` + 5-Fold Cross-Validation |

---

## 🤖 Modèles entraînés

### Régression (prédire le prix)
| Modèle | R² | RMSE |
|---|---|---|
| Régression Linéaire | ~0.879 | ~2179$ |
| Arbre de Régression | ~0.953 | ~1355$ |
| SVR | ~0.940 | ~1510$ |
| **Random Forest** ⭐ | **~0.965** | **~1150$** |

### Classification (prix haut de gamme ?)
| Modèle | Accuracy | F1-Score |
|---|---|---|
| Régression Logistique | ~0.940 | ~0.935 |
| Arbre de Décision | ~0.945 | ~0.940 |
| SVC | ~0.942 | ~0.937 |
| **Random Forest** ⭐ | **~0.955** | **~0.952** |

---

## 📁 Structure du projet

```
diamonds-ml/
├── app.py                      # Application Streamlit
├── Diamonds_ML_complet.ipynb   # Notebook complet (pipeline ML)
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

---

## ⚙️ Installation & Lancement

```bash
# 1. Cloner le dépôt
git clone https://github.com/cesarlome4/diamonds-ml.git
cd diamonds-ml

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'application
streamlit run app.py
```

---

## 🧪 Technologies utilisées

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)

- **Python** — langage principal
- **Streamlit** — interface web interactive
- **scikit-learn** — modèles ML, preprocessing, évaluation
- **Pandas / NumPy** — manipulation des données
- **Matplotlib / Seaborn** — visualisations

---

## 💎 Features du dataset

| Feature | Type | Description |
|---|---|---|
| `carat` | Numérique | Poids du diamant |
| `cut` | Catégorielle ordinale | Qualité de la taille (Fair → Ideal) |
| `color` | Catégorielle ordinale | Couleur (J → D) |
| `clarity` | Catégorielle ordinale | Clarté (I1 → IF) |
| `depth` | Numérique | Profondeur en % |
| `table` | Numérique | Largeur de la table en % |
| `x`, `y`, `z` | Numérique | Dimensions en mm |
| `price` | Numérique | **Cible** — prix en $ |

---

## 👤 Auteur

**Cesario Lome**  
[![GitHub](https://img.shields.io/badge/GitHub-cesarlome4-black?logo=github)](https://github.com/cesarlome4)
