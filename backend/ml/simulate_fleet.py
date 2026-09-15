"""
STARGUARD AI — симуляция парка изделий (цифровые двойники) для демо.

Логика связки идей 1+3: для каждого serial number сначала генерируется
QC-результат производства (реальный инференс CNN на synthetic-дефектах),
затем — телеметрия эксплуатации (СМАРТ ТМ/АВР: температура токоведущих
частей, ток нагрузки). Юниты с худшим Quality Score на производстве получают
повышенную вероятность деградации в эксплуатации — это и есть замыкание
контура "качество производства -> надёжность в поле", ключевая идея продукта.
"""
import os
import json
import random
import subprocess
from glob import glob

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
from defect_model import analyze_image  # noqa: E402
from predictive_model import predict_early_warning  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data", "synthetic_defects")
OUT_DIR = os.path.join(HERE, "..", "data", "fleet")
os.makedirs(OUT_DIR, exist_ok=True)

CPP_ENGINE = os.path.join(HERE, "..", "cpp", "build", "telemetry_engine")


def run_cpp_engine(days, temps, currents):
    """Прогоняет телеметрию юнита через C++ движок потокового детектора аномалий."""
    lines = "\n".join(f"{d},{t},{c}" for d, t, c in zip(days, temps, currents))
    try:
        proc = subprocess.run([CPP_ENGINE], input=lines, capture_output=True, text=True, timeout=10)
        return json.loads(proc.stdout.strip())
    except Exception as e:  # noqa: BLE001
        return {"status": "UNKNOWN", "error": str(e)}

CUSTOMERS = [
    {"name": "ПАО «Лукойл» — Прохоровское м/р, Усинск", "product": "КТП 6/0.4 кВ, 1000 кВА"},
    {"name": "Лукойл — Западная Сибирь, Покамасовское м/р", "product": "Подстанция 35 кВ"},
    {"name": "Ташкентская ТЭС, энергоблок 368 МВт", "product": "РУСН-1 2х5000А"},
    {"name": "Газпромнефть — НПЗ", "product": "НКУ «Протон» 0.4 кВ"},
    {"name": "Курская АЭС — вспомогательные системы", "product": "КРУ «Позитрон» 10 кВ"},
    {"name": "Промышленный узел, Екатеринбург", "product": "КРУЭ SYSTEMEGT 20 кВ"},
    {"name": "Металлургический комбинат, Челябинск", "product": "НКУ с ЧП до 220 кВт"},
    {"name": "Нефтехимический завод, Тобольск", "product": "2ТП-4000кВА"},
]

random.seed(7)
np.random.seed(7)

N_DAYS = 180


def pick_qc_image(cls, rng):
    files = sorted(glob(os.path.join(DATA_DIR, cls, "*.png")))
    return rng.choice(files)


def gen_serial(i):
    return f"ENS-{2024 + i // 40}-{10000 + i}"


def simulate_telemetry(quality_score, rng):
    """Генерирует 180 дней телеметрии (температура токоведущих частей, °C; ток, А).
    Чем ниже quality_score (хуже произведён узел), тем выше вероятность и скорость
    деградации (дрейф температуры, рост дисперсии) в течение периода эксплуатации."""
    base_temp = rng.uniform(38, 46)
    base_current = rng.uniform(180, 420)

    degradation_risk = np.clip((100 - quality_score) / 100, 0, 1)
    will_degrade = rng.random() < (0.12 + 0.65 * degradation_risk)

    temps = []
    currents = []
    degrade_start = int(N_DAYS * rng.uniform(0.55, 0.85)) if will_degrade else None
    drift_rate = rng.uniform(0.15, 0.45) if will_degrade else 0.0

    t = base_temp
    for day in range(N_DAYS):
        noise = rng.normal(0, 0.6)
        drift = 0.0
        if will_degrade:
            precursor = 2.0 * drift_rate * day / N_DAYS
            main = drift_rate * (day - degrade_start) ** 1.15 / 10 if day >= degrade_start else 0.0
            drift = precursor + main
        t_today = base_temp + noise + drift
        temps.append(round(float(t_today), 2))

        c_noise = rng.normal(0, 8)
        c_today = base_current + c_noise + (drift * 2 if will_degrade and day >= degrade_start else 0)
        currents.append(round(float(max(0, c_today)), 1))

    return {
        "days": list(range(N_DAYS)),
        "temperature_c": temps,
        "current_a": currents,
        "will_degrade": bool(will_degrade),
        "degrade_start_day": degrade_start,
    }


def build_fleet(n_units=24):
    units = []
    classes = ["ok"] * 10 + ["scratches"] * 4 + ["pitted_surface"] * 3 + ["inclusion"] * 3 + \
              ["patches"] * 2 + ["rolled_in_scale"] * 1 + ["crazing"] * 1
    rng_master = np.random.default_rng(123)

    for i in range(n_units):
        rng = np.random.default_rng(1000 + i)
        cls = classes[i % len(classes)]
        img_path = pick_qc_image(cls, random.Random(2000 + i))
        qc = analyze_image(img_path)
        customer = CUSTOMERS[i % len(CUSTOMERS)]
        telemetry = simulate_telemetry(qc["quality_score"], rng)

        cpp_result = run_cpp_engine(telemetry["days"], telemetry["temperature_c"], telemetry["current_a"])
        ai_prob = predict_early_warning(telemetry["temperature_c"], telemetry["current_a"])

        cpp_anomaly_day = cpp_result.get("anomaly_start_day", -1)
        lead_time_days = None
        if cpp_anomaly_day and cpp_anomaly_day > 90:
            lead_time_days = cpp_anomaly_day - 90  # насколько раньше AI мог бы предупредить vs классика

        if cpp_result.get("status") == "CRITICAL":
            health_status = "CRITICAL"
        elif cpp_result.get("status") == "WARNING" or (ai_prob is not None and ai_prob >= 0.65):
            health_status = "WARNING"
        else:
            health_status = "OK"

        units.append({
            "serial": gen_serial(i),
            "customer": customer["name"],
            "product": customer["product"],
            "manufactured_date": f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "qc": {
                "image_path": os.path.relpath(img_path, os.path.join(HERE, "..")),
                "predicted_class": qc["predicted_class"],
                "confidence": qc["confidence"],
                "quality_score": qc["quality_score"],
                "probabilities": qc["probabilities"],
            },
            "telemetry": telemetry,
            "cpp_realtime_analysis": cpp_result,
            "ai_early_warning": {
                "probability": ai_prob,
                "observed_days": 90,
            },
            "health_status": health_status,
            "lead_time_days_vs_classic": lead_time_days,
        })
    return units


if __name__ == "__main__":
    fleet = build_fleet()
    with open(os.path.join(OUT_DIR, "fleet.json"), "w", encoding="utf-8") as f:
        json.dump(fleet, f, ensure_ascii=False, indent=2)
    n_degrade = sum(1 for u in fleet if u["telemetry"]["will_degrade"])
    print(f"Сгенерирован парк из {len(fleet)} изделий, из них с признаками деградации: {n_degrade}")
