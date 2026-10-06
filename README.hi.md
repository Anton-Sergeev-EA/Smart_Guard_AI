# Smart_Guard_AI

[Русский](README.md) · [English](README.en.md) · [中文](README.zh.md) · **हिन्दी** · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md)

Serial number से manufacturing quality inspection और condition monitoring जोड़ने वाला laboratory PoC। अतिरिक्त portfolio project; अभी feature expansion नहीं। Images और telemetry synthetic हैं; industrial deployment और business benefit सिद्ध नहीं।

![Smart_Guard_AI](docs/screenshot_dashboard.png)

## आर्किटेक्चर

PyTorch CNN procedural surface defects वर्गीकृत करता है और Quality Score देता है। LSTM prototype telemetry और quality-related inputs लेता है। C++17 executable telemetry process करता है; FastAPI digital-passport fleet और HTML/CSS/JS dashboard देता है। Explanations local retrieval और templates हैं, external LLM नहीं।

## चलाना और सत्यापन

Repository root से commands चलाएँ। Python dependencies पूरी तरह pinned नहीं। Training और fleet generation generated artifacts overwrite करते हैं; पुराने models बचाने के लिए अलग checkout उपयोग करें। Dashboard: http://localhost:8000।

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cmake -S backend/cpp -B backend/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/cpp/build -j2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

## वैकल्पिक पुनः निर्माण

```sh
cd backend/ml
../../.venv/bin/python generate_dataset.py
../../.venv/bin/python train_defect_model.py
../../.venv/bin/python train_predictive_model.py
../../.venv/bin/python simulate_fleet.py
```

## साक्ष्य की सीमाएँ

Quality Score और future failure का संबंध research hypothesis है। Accuracy, कम downtime, कम claims और financial returns के लिए independent real-data evaluation चाहिए। Demo site names installations सिद्ध नहीं करते। lead_time_days_vs_classic threshold day से fixed day 90 घटाता है; यह पहला sequential model alert नहीं मापता। Seeds Python hash() की जगह SHA-256 हैं; retraining/evaluation बिना पुराने weights/metrics बदले dataset पर लागू नहीं। C++ --bench केवल in-memory loop मापता है, collection-to-dashboard throughput या industrial edge नहीं।

## दस्तावेज़

ये concise localized guides हैं; detailed examples Russian README में हैं। UI और generated explanations localize नहीं हुए। Normal-input smoke और reproducibility test CNN/LSTM accuracy, RUL या production recovery validate नहीं करते।

[Русский](README.md)
