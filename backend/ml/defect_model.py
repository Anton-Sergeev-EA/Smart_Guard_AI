"""
STARGUARD AI — инференс CNN-классификатора дефектов поверхности.
"""
import os
import numpy as np
import torch
from PIL import Image

from train_defect_model import DefectCNN
from generate_dataset import SEVERITY

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "data", "models", "defect_cnn.pt")

_cache = {}


def _load():
    if "model" in _cache:
        return _cache["model"], _cache["classes"], _cache["img_size"]
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt["img_size"]
    model = DefectCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _cache.update(model=model, classes=classes, img_size=img_size)
    return model, classes, img_size


def analyze_image(path_or_pil):
    """Возвращает предсказанный класс дефекта, уверенность, распределение вероятностей
    и итоговый Quality Score (0-100, где 100 = идеальная поверхность)."""
    model, classes, img_size = _load()
    if isinstance(path_or_pil, (str, os.PathLike)):
        img = Image.open(path_or_pil).convert("L")
    else:
        img = path_or_pil.convert("L")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.25
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    # Quality score: штраф пропорционален severity дефекта и уверенности модели в нём
    severity = SEVERITY.get(pred_class, 0.0)
    penalty = severity * confidence * 100
    quality_score = max(0.0, 100.0 - penalty)

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "quality_score": round(quality_score, 1),
        "severity": severity,
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


if __name__ == "__main__":
    import sys
    print(analyze_image(sys.argv[1]))
