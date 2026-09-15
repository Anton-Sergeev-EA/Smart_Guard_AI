"""
STARGUARD AI — генератор синтетического датасета дефектов поверхности металла.

В продакшене этот модуль заменяется реальным потоком с камер на постах лазерной
резки / сварки / порошковой покраски. Для PoC мы процедурно синтезируем текстуры,
имитирующие типовые дефекты стальной поверхности (по мотивам классической
таксономии NEU surface-defect: crazing, inclusion, patches, pitted_surface,
rolled-in_scale, scratches), плюс класс "ok" — бездефектная поверхность.

Это даёт полноценный, воспроизводимый датасет для обучения настоящей CNN без
зависимости от внешних источников данных.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMG_SIZE = 128
CLASSES = ["ok", "scratches", "pitted_surface", "inclusion", "patches", "rolled_in_scale", "crazing"]
SEVERITY = {  # вес дефекта для расчёта Quality Score (0 = не влияет, 1 = максимально критично)
    "ok": 0.0,
    "scratches": 0.35,
    "pitted_surface": 0.55,
    "inclusion": 0.65,
    "patches": 0.30,
    "rolled_in_scale": 0.45,
    "crazing": 0.80,
}

rng = np.random.default_rng(42)


def base_steel_texture(rng_local):
    """Базовая текстура шлифованной стали: направленный шум + лёгкий градиент освещения."""
    base = 150 + rng_local.normal(0, 6, (IMG_SIZE, IMG_SIZE))
    # направленные полосы шлифовки
    for _ in range(40):
        y = rng_local.integers(0, IMG_SIZE)
        x0 = rng_local.integers(0, IMG_SIZE // 4)
        length = rng_local.integers(IMG_SIZE // 2, IMG_SIZE)
        thickness = rng_local.choice([1, 1, 2])
        val = rng_local.normal(0, 2)
        y0, y1 = max(0, y - thickness), min(IMG_SIZE, y + thickness)
        x1 = min(IMG_SIZE, x0 + length)
        base[y0:y1, x0:x1] += val
    # лёгкий градиент освещения
    grad = np.linspace(-8, 8, IMG_SIZE).reshape(1, -1)
    base += grad
    return np.clip(base, 0, 255)


def add_scratches(arr, rng_local, n=None):
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    draw = ImageDraw.Draw(img)
    n = n or rng_local.integers(2, 6)
    for _ in range(n):
        x0, y0 = rng_local.integers(0, IMG_SIZE, 2)
        length = rng_local.integers(20, 90)
        angle = rng_local.uniform(0, np.pi)
        x1 = int(np.clip(x0 + length * np.cos(angle), 0, IMG_SIZE - 1))
        y1 = int(np.clip(y0 + length * np.sin(angle), 0, IMG_SIZE - 1))
        shade = int(rng_local.uniform(40, 90))
        draw.line([(x0, y0), (x1, y1)], fill=shade, width=1)
    return np.array(img, dtype=np.float64)


def add_pitting(arr, rng_local, n=None):
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    draw = ImageDraw.Draw(img)
    n = n or rng_local.integers(15, 40)
    for _ in range(n):
        x, y = rng_local.integers(5, IMG_SIZE - 5, 2)
        r = rng_local.integers(1, 4)
        shade = int(rng_local.uniform(50, 100))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=shade)
    return np.array(img, dtype=np.float64)


def add_inclusion(arr, rng_local, n=None):
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    draw = ImageDraw.Draw(img)
    n = n or rng_local.integers(3, 8)
    for _ in range(n):
        cx, cy = rng_local.integers(15, IMG_SIZE - 15, 2)
        pts = []
        rpts = rng_local.integers(6, 12)
        for k in range(rpts):
            ang = 2 * np.pi * k / rpts
            rad = rng_local.uniform(4, 14)
            pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
        shade = int(rng_local.uniform(200, 240)) if rng_local.random() > 0.5 else int(rng_local.uniform(30, 70))
        draw.polygon(pts, fill=shade)
    return np.array(img, dtype=np.float64)


def add_patches(arr, rng_local, n=None):
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    draw = ImageDraw.Draw(img)
    n = n or rng_local.integers(1, 3)
    for _ in range(n):
        cx, cy = rng_local.integers(20, IMG_SIZE - 20, 2)
        rw, rh = rng_local.integers(15, 35, 2)
        shade = int(rng_local.uniform(170, 210))
        draw.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=shade)
    out = np.array(img, dtype=np.float64)
    return out


def add_rolled_in_scale(arr, rng_local, n=None):
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    draw = ImageDraw.Draw(img)
    n = n or rng_local.integers(4, 10)
    for _ in range(n):
        x0, y0 = rng_local.integers(0, IMG_SIZE, 2)
        length = rng_local.integers(10, 40)
        angle = rng_local.uniform(0, np.pi)
        x1 = int(np.clip(x0 + length * np.cos(angle), 0, IMG_SIZE - 1))
        y1 = int(np.clip(y0 + length * np.sin(angle), 0, IMG_SIZE - 1))
        shade = int(rng_local.uniform(60, 110))
        draw.line([(x0, y0), (x1, y1)], fill=shade, width=rng_local.integers(2, 4))
    return np.array(img, dtype=np.float64)


def add_crazing(arr, rng_local):
    img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    draw = ImageDraw.Draw(img)

    def branch(x, y, angle, depth):
        if depth <= 0:
            return
        length = rng_local.uniform(6, 16)
        x1 = x + length * np.cos(angle)
        y1 = y + length * np.sin(angle)
        if 0 <= x1 < IMG_SIZE and 0 <= y1 < IMG_SIZE:
            draw.line([(x, y), (x1, y1)], fill=int(rng_local.uniform(60, 100)), width=1)
            if rng_local.random() > 0.35:
                branch(x1, y1, angle + rng_local.uniform(-0.6, 0.6), depth - 1)
            if rng_local.random() > 0.7:
                branch(x1, y1, angle + rng_local.uniform(-1.2, 1.2), depth - 1)

    for _ in range(rng_local.integers(3, 6)):
        x0, y0 = rng_local.integers(10, IMG_SIZE - 10, 2)
        branch(x0, y0, rng_local.uniform(0, 2 * np.pi), depth=6)
    return np.array(img, dtype=np.float64)


DEFECT_FN = {
    "scratches": add_scratches,
    "pitted_surface": add_pitting,
    "inclusion": add_inclusion,
    "patches": add_patches,
    "rolled_in_scale": add_rolled_in_scale,
    "crazing": add_crazing,
}


def make_image(cls, seed):
    rng_local = np.random.default_rng(seed)
    arr = base_steel_texture(rng_local)
    if cls != "ok":
        arr = DEFECT_FN[cls](arr, rng_local)
    arr = np.clip(arr + rng_local.normal(0, 3, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(0.4))
    return img


def generate(out_dir, n_per_class=320, seed_offset=0):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for cls in CLASSES:
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            seed = seed_offset + hash((cls, i)) % (2**31)
            img = make_image(cls, seed)
            path = os.path.join(cls_dir, f"{cls}_{i:04d}.png")
            img.save(path)
            manifest.append({"path": path, "class": cls})
    return manifest


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_defects")
    print("Генерация синтетического датасета дефектов поверхности стали...")
    m = generate(base, n_per_class=320)
    print(f"Готово: {len(m)} изображений, классы: {CLASSES}")
