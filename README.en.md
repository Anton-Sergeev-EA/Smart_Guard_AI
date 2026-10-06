# Smart_Guard_AI

[Русский](README.md) · **English** · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md)

Laboratory proof of concept connecting manufacturing quality inspection with condition monitoring by serial number. Additional portfolio project; no feature expansion is currently planned. Images and telemetry are synthetic; industrial deployment and measured business benefit are not established.

![Smart_Guard_AI](docs/screenshot_dashboard.png)

## Architecture

PyTorch CNN classifies procedural surface defects and produces Quality Score. The LSTM prototype uses telemetry and quality-related inputs. A C++17 executable processes telemetry; FastAPI serves the digital-passport fleet and HTML/CSS/JS dashboard. Explanations use local retrieval plus templates, not an external LLM.

## Run and verify

Run these commands from the repository root. Python dependencies are not fully pinned. Training and fleet generation overwrite generated artifacts; perform them in a separate checkout when preserving existing models. Dashboard: http://localhost:8000.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cmake -S backend/cpp -B backend/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/cpp/build -j2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

## Optional regeneration

```sh
cd backend/ml
../../.venv/bin/python generate_dataset.py
../../.venv/bin/python train_defect_model.py
../../.venv/bin/python train_predictive_model.py
../../.venv/bin/python simulate_fleet.py
```

## Evidence boundaries

The Quality Score–future failure relationship is a research hypothesis. Accuracy, lower downtime, fewer claims and financial returns need independent real-data evaluation. Demo site names do not prove installations. lead_time_days_vs_classic subtracts fixed day 90 from the threshold day; it does not measure the first sequential model alert. Dataset seeds use SHA-256 instead of Python hash(); existing weights and older metrics cannot be assigned to the changed dataset without retraining and evaluation. C++ --bench measures an in-memory computation loop, not collection-to-dashboard throughput or industrial edge operation.

## Documentation

These are concise localized guides; detailed examples remain in the Russian README. Application UI and generated explanations are not localized by this change. The normal-input smoke and image reproducibility test do not validate CNN/LSTM accuracy, RUL or production recovery.

[Русский](README.md)
