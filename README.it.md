# Smart_Guard_AI

[Русский](README.md) · [English](README.en.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · **Italiano**

PoC di laboratorio che collega controllo qualità e monitoraggio tramite numero di serie. Progetto complementare senza ampliamento previsto. Immagini e telemetria sintetiche; deployment industriale e benefici non dimostrati.

![Smart_Guard_AI](docs/screenshot_dashboard.png)

## Architettura

La CNN PyTorch classifica difetti procedurali e produce Quality Score. Il prototipo LSTM usa telemetria e input legati alla qualità. Un eseguibile C++17 elabora telemetria; FastAPI serve passaporti digitali e dashboard HTML/CSS/JS. Le spiegazioni usano recupero locale e modelli di testo, non un LLM esterno.

## Avvio e verifica

Eseguire dalla radice del repository. Le dipendenze Python non sono completamente fissate. Training e generazione della flotta sovrascrivono artefatti; usare una copia separata per conservare i modelli. Dashboard: http://localhost:8000.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cmake -S backend/cpp -B backend/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/cpp/build -j2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

## Rigenerazione opzionale

```sh
cd backend/ml
../../.venv/bin/python generate_dataset.py
../../.venv/bin/python train_defect_model.py
../../.venv/bin/python train_predictive_model.py
../../.venv/bin/python simulate_fleet.py
```

## Limiti delle prove

Il legame Quality Score–guasto futuro è un’ipotesi. Accuratezza, minori fermi/reclami e benefici economici richiedono valutazione indipendente su dati reali. I nomi dei siti non provano installazioni. lead_time_days_vs_classic sottrae il giorno fisso 90 dal giorno della soglia; non misura il primo allarme sequenziale. I seed usano SHA-256 invece di Python hash(); vecchi pesi e metriche non appartengono al dataset modificato senza nuovo training e valutazione. C++ --bench misura un ciclo in memoria, non throughput completo o impiego edge industriale.

## Documentazione

Guide localizzate concise; esempi dettagliati nel README russo. Interfaccia e spiegazioni non sono tradotte. Smoke normale e riproducibilità delle immagini non validano accuratezza CNN/LSTM, RUL o ripristino in produzione.

[Русский](README.md)
