# Smart_Guard_AI

[Русский](README.md) · [English](README.en.md) · [中文](README.zh.md) · [हिन्दी](README.hi.md) · **Español** · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md)

Prueba de concepto de laboratorio que vincula inspección de calidad y monitorización por número de serie. Proyecto complementario; no se amplían funciones. Imágenes y telemetría sintéticas; despliegue industrial y beneficios no demostrados.

![Smart_Guard_AI](docs/screenshot_dashboard.png)

## Arquitectura

La CNN PyTorch clasifica defectos procedurales y produce Quality Score. El prototipo LSTM utiliza telemetría y entradas de calidad. Un ejecutable C++17 procesa telemetría; FastAPI sirve la flota de pasaportes digitales y el panel HTML/CSS/JS. Las explicaciones usan recuperación local y plantillas, no un LLM externo.

## Ejecución y verificación

Ejecute desde la raíz del repositorio. Las dependencias Python no están totalmente fijadas. Entrenamiento y generación de flota sobrescriben artefactos; use una copia separada para conservar modelos. Panel: http://localhost:8000.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cmake -S backend/cpp -B backend/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/cpp/build -j2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

## Regeneración opcional

```sh
cd backend/ml
../../.venv/bin/python generate_dataset.py
../../.venv/bin/python train_defect_model.py
../../.venv/bin/python train_predictive_model.py
../../.venv/bin/python simulate_fleet.py
```

## Límites de evidencia

La relación Quality Score–fallo futuro es una hipótesis. Precisión, reducción de paradas/reclamaciones y rentabilidad requieren evaluación independiente con datos reales. Los nombres de sitios no prueban instalaciones. lead_time_days_vs_classic resta el día fijo 90 del día del umbral; no mide la primera alerta secuencial. Las semillas usan SHA-256, no Python hash(); pesos y métricas anteriores no corresponden al dataset cambiado sin reentrenamiento y evaluación. C++ --bench mide un bucle en memoria, no el rendimiento completo ni equipos edge industriales.

## Documentación

Guías localizadas breves; ejemplos detallados en el README ruso. No traducen interfaz ni explicaciones. Smoke con entradas normales y reproducibilidad de imágenes no validan precisión CNN/LSTM, RUL ni recuperación en producción.

[Русский](README.md)
