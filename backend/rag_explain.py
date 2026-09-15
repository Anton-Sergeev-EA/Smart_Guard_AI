"""
STARGUARD AI — RAG-объяснение алертов.

Реализован настоящий retrieval-слой (поиск релевантных записей локальной базы
эксплуатационных знаний) + шаблонная генерация связного объяснения на русском
языке с подстановкой конкретных цифр юнита. В продакшене шаг генерации
заменяется вызовом LLM (Claude/GPT) поверх тех же retrieved-фрагментов —
retrieval и есть содержательная часть RAG, которую мы здесь честно
демонстрируем без обращения к внешним API.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "knowledge_base"))
from guidelines import retrieve  # noqa: E402


def explain_unit(unit: dict) -> dict:
    qc = unit["qc"]
    cpp = unit.get("cpp_realtime_analysis", {})
    ai = unit.get("ai_early_warning", {})
    status = unit.get("health_status", "OK")

    defect_class = qc["predicted_class"] if qc["predicted_class"] != "ok" else None
    docs = retrieve(defect_class=defect_class, extra_query=status)

    parts = []
    parts.append(
        f"Изделие {unit['serial']} ({unit['product']}, заказчик: {unit['customer']}) "
        f"прошло выходной контроль с результатом «{qc['predicted_class']}» "
        f"(уверенность модели {qc['confidence']*100:.1f}%, Quality Score {qc['quality_score']:.0f}/100)."
    )

    if defect_class:
        kb_defect = next((d for d in docs if d["defect_class"] == defect_class), None)
        if kb_defect:
            parts.append(kb_defect["text"])
    else:
        parts.append("При выходном контроле дефектов поверхности не выявлено.")

    if status in ("WARNING", "CRITICAL"):
        if cpp.get("status") in ("WARNING", "CRITICAL"):
            day = cpp.get("anomaly_start_day", -1)
            if day and day > 0:
                parts.append(
                    f"Потоковый анализ телеметрии (C++ движок) зафиксировал устойчивое "
                    f"отклонение температуры токоведущих частей от пуско-наладочной базовой "
                    f"линии начиная примерно с {day}-х суток эксплуатации."
                )
            else:
                parts.append(
                    "Потоковый анализ телеметрии показывает нарастающее отклонение от "
                    "базовой линии, устойчивого превышения порога тревоги пока не "
                    "зафиксировано."
                )
        prob = ai.get("probability")
        if prob is not None and prob >= 0.5:
            lead = unit.get("lead_time_days_vs_classic")
            lead_txt = (
                f" — это на {lead} суток раньше, чем сработал бы классический пороговый детектор."
                if lead else ""
            )
            parts.append(
                f"Нейросеть раннего предупреждения оценивает вероятность будущей "
                f"эксплуатационной деградации в {prob*100:.0f}% уже по первым 90 суткам "
                f"телеметрии{lead_txt}"
            )
        kb_general = next((d for d in docs if d["id"] == "kb_thermal_general"), None)
        if kb_general:
            parts.append(kb_general["text"])
    else:
        parts.append("Признаков эксплуатационной деградации по телеметрии не выявлено, состояние в норме.")

    if defect_class and status in ("WARNING", "CRITICAL"):
        kb_link = next((d for d in docs if d["id"] == "kb_quality_link"), None)
        if kb_link:
            parts.append(kb_link["text"])

    return {
        "serial": unit["serial"],
        "status": status,
        "explanation": " ".join(parts),
        "sources": [d["id"] for d in docs if d in docs][:4],
    }


if __name__ == "__main__":
    import json
    HERE = os.path.dirname(__file__)
    fleet = json.load(open(os.path.join(HERE, "data", "fleet", "fleet.json"), encoding="utf-8"))
    for u in fleet:
        if u["health_status"] != "OK":
            print(json.dumps(explain_unit(u), ensure_ascii=False, indent=2))
            print("---")
