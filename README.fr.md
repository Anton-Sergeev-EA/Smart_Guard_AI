# Smart_Guard_AI

[Русский](README.md) · [English](README.en.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · **Français** · [Deutsch](README.de.md) · [Italiano](README.it.md)

Preuve de concept de laboratoire reliant contrôle qualité et surveillance par numéro de série. Projet complémentaire, sans extension prévue. Images et télémétrie synthétiques ; déploiement industriel et bénéfices non démontrés.

![Smart_Guard_AI](docs/screenshot_dashboard.png)

## Architecture

La CNN PyTorch classe des défauts procéduraux et produit Quality Score. Le prototype LSTM utilise télémétrie et entrées liées à la qualité. Un exécutable C++17 traite la télémétrie ; FastAPI sert les passeports numériques et le tableau HTML/CSS/JS. Explications par recherche locale et modèles de texte, sans LLM externe.

## Exécution et vérification

Exécutez depuis la racine du dépôt. Les dépendances Python ne sont pas entièrement figées. Entraînement et génération de flotte écrasent les artefacts ; utilisez une copie distincte pour conserver les modèles. Tableau : http://localhost:8000.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cmake -S backend/cpp -B backend/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/cpp/build -j2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

## Régénération facultative

```sh
cd backend/ml
../../.venv/bin/python generate_dataset.py
../../.venv/bin/python train_defect_model.py
../../.venv/bin/python train_predictive_model.py
../../.venv/bin/python simulate_fleet.py
```

## Limites des preuves

Le lien Quality Score–panne future est une hypothèse. Précision, réduction des arrêts/réclamations et gains exigent une évaluation indépendante sur données réelles. Les noms des sites ne prouvent aucune installation. lead_time_days_vs_classic soustrait le jour fixe 90 au jour du seuil ; il ne mesure pas la première alerte séquentielle. Les seeds utilisent SHA-256 et non Python hash() ; anciens poids et métriques ne correspondent pas au nouveau dataset sans réentraînement et évaluation. C++ --bench mesure une boucle en mémoire, pas le débit complet ni une utilisation edge industrielle.

## Documentation

Guides localisés concis ; exemples détaillés dans le README russe. Interface et explications ne sont pas traduites. Smoke normal et reproductibilité des images ne valident ni précision CNN/LSTM, ni RUL, ni reprise en production.

[Русский](README.md)
