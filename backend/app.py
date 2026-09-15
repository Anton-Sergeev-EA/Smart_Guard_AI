"""
STARGUARD AI — FastAPI backend.

Собирает воедино: CNN-контроль качества производства (ml/defect_model.py),
LSTM раннего предупреждения (ml/predictive_model.py), потоковый C++ движок
телеметрии (cpp/build/telemetry_engine), RAG-объяснения (rag_explain.py) и
отдаёт всё дашборду (frontend/) как единый цифровой двойник парка изделий.
"""
import os
import sys
import json
import subprocess
import shutil
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ml"))
sys.path.insert(0, HERE)

from defect_model import analyze_image  # noqa: E402
from rag_explain import explain_unit  # noqa: E402

FLEET_PATH = os.path.join(HERE, "data", "fleet", "fleet.json")
CPP_ENGINE = os.path.join(HERE, "cpp", "build", "telemetry_engine")
FRONTEND_DIR = os.path.join(HERE, "..", "frontend")
DATA_DIR = os.path.join(HERE, "data")

app = FastAPI(title="STARGUARD AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_fleet():
    with open(FLEET_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/units")
def list_units():
    fleet = load_fleet()
    summary = []
    for u in fleet:
        summary.append({
            "serial": u["serial"],
            "customer": u["customer"],
            "product": u["product"],
            "manufactured_date": u["manufactured_date"],
            "quality_score": u["qc"]["quality_score"],
            "defect_class": u["qc"]["predicted_class"],
            "health_status": u["health_status"],
            "ai_probability": u["ai_early_warning"]["probability"],
        })
    return {
        "count": len(summary),
        "critical": sum(1 for s in summary if s["health_status"] == "CRITICAL"),
        "warning": sum(1 for s in summary if s["health_status"] == "WARNING"),
        "ok": sum(1 for s in summary if s["health_status"] == "OK"),
        "units": summary,
    }


@app.get("/api/units/{serial}")
def get_unit(serial: str):
    fleet = load_fleet()
    unit = next((u for u in fleet if u["serial"] == serial), None)
    if not unit:
        raise HTTPException(status_code=404, detail="unit not found")
    return unit


@app.get("/api/units/{serial}/explain")
def get_unit_explanation(serial: str):
    fleet = load_fleet()
    unit = next((u for u in fleet if u["serial"] == serial), None)
    if not unit:
        raise HTTPException(status_code=404, detail="unit not found")
    return explain_unit(unit)


@app.get("/api/units/{serial}/qc-image")
def get_unit_qc_image(serial: str):
    fleet = load_fleet()
    unit = next((u for u in fleet if u["serial"] == serial), None)
    if not unit:
        raise HTTPException(status_code=404, detail="unit not found")
    path = os.path.join(HERE, unit["qc"]["image_path"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path)


@app.post("/api/qc/analyze")
async def qc_analyze(file: UploadFile = File(...)):
    """Живой демо-инференс: загрузить изображение поверхности -> результат CNN."""
    content = await file.read()
    try:
        img = Image.open(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")
    result = analyze_image(img)
    return result


@app.get("/api/qc/sample-images")
def list_sample_images():
    """Список готовых демо-изображений (по одному на класс) для живой демонстрации
    без необходимости заранее иметь файлы дефектов под рукой."""
    base = os.path.join(DATA_DIR, "synthetic_defects")
    out = []
    for cls in sorted(os.listdir(base)):
        cls_dir = os.path.join(base, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(os.listdir(cls_dir))
        if files:
            out.append({"class": cls, "path": f"/api/qc/sample-image/{cls}/{files[0]}"})
    return out


@app.get("/api/qc/sample-image/{cls}/{fname}")
def get_sample_image(cls: str, fname: str):
    path = os.path.join(DATA_DIR, "synthetic_defects", cls, fname)
    if not os.path.exists(path) or not os.path.abspath(path).startswith(os.path.abspath(DATA_DIR)):
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


@app.get("/api/units/{serial}/realtime-replay")
def realtime_replay(serial: str, up_to_day: int = 180):
    """Прогоняет телеметрию юнита через C++ движок только до дня up_to_day —
    имитация 'живого' поступления данных для демонстрации потоковой обработки."""
    fleet = load_fleet()
    unit = next((u for u in fleet if u["serial"] == serial), None)
    if not unit:
        raise HTTPException(status_code=404, detail="unit not found")
    t = unit["telemetry"]
    days = t["days"][:up_to_day]
    temps = t["temperature_c"][:up_to_day]
    currents = t["current_a"][:up_to_day]
    lines = "\n".join(f"{d},{te},{c}" for d, te, c in zip(days, temps, currents))
    try:
        proc = subprocess.run([CPP_ENGINE], input=lines, capture_output=True, text=True, timeout=10)
        result = json.loads(proc.stdout.strip())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/api/engine/benchmark")
def engine_benchmark(n: int = 1_000_000):
    n = min(max(n, 1000), 20_000_000)
    try:
        proc = subprocess.run([CPP_ENGINE, "--bench", str(n)], capture_output=True, text=True, timeout=30)
        return json.loads(proc.stdout.strip())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "cpp_engine_present": os.path.exists(CPP_ENGINE),
        "fleet_size": len(load_fleet()),
    }


# Статика фронтенда — раздаём собранный дашборд той же FastAPI-инстанцией,
# чтобы всё демо поднималось одной командой.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
