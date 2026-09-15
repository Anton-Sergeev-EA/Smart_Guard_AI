"""
STARGUARD AI — LSTM-модель раннего предупреждения об эксплуатационной деградации.

Идея: пороговый/статистический детектор (C++ telemetry_engine) реагирует
ПОСТФАКТУМ — когда аномалия уже устойчиво проявилась. Нейросеть учится
предсказывать риск будущей деградации, видя только ПЕРВЫЕ 90 дней
эксплуатации (тренд + характер шума), то есть даёт задел по времени для
превентивного обслуживания. Обучается на отдельном синтетическом наборе
временных рядов (не пересекается с демо-парком) с известной разметкой.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

N_DAYS_TOTAL = 180
OBS_WINDOW = 90  # модель видит только первые 90 дней


def simulate_series(rng, degrade_risk):
    base_temp = rng.uniform(38, 46)
    base_current = rng.uniform(180, 420)
    will_degrade = rng.random() < degrade_risk
    degrade_start = int(N_DAYS_TOTAL * rng.uniform(0.55, 0.85)) if will_degrade else None
    drift_rate = rng.uniform(0.15, 0.45) if will_degrade else 0.0

    temps, currents = [], []
    for day in range(N_DAYS_TOTAL):
        noise = rng.normal(0, 0.6)
        drift = 0.0
        if will_degrade:
            # слабый, но статистически устойчивый "предвестник" деградации с самого
            # начала эксплуатации (микро-тренд, незаметный порогу в sigma, но видимый
            # по форме тренда на длинном окне) + основной дрейф после видимого начала
            precursor = 2.0 * drift_rate * day / N_DAYS_TOTAL
            main = drift_rate * (day - degrade_start) ** 1.15 / 10 if day >= degrade_start else 0.0
            drift = precursor + main
        temps.append(base_temp + noise + drift)
        c_noise = rng.normal(0, 8)
        currents.append(max(0, base_current + c_noise + (drift * 2 if will_degrade and day >= degrade_start else 0)))
    return np.array(temps), np.array(currents), will_degrade, degrade_start


def rolling_mean(arr, window=5):
    kernel = np.ones(window) / window
    # 'same'-паддинг через отражение краёв, чтобы не терять длину окна наблюдения
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def trend_features(obs):
    """Явные признаки тренда на окне: наклон линейной регрессии, разница
    последних/первых 10 точек, дисперсия. Это ровно то, что физически ищет
    инженер, глядя на график — даём сети эти же подсказки явно, а не только
    сырые зашумлённые точки."""
    days = np.arange(len(obs))
    slope = np.polyfit(days, obs, 1)[0]
    diff = obs[-10:].mean() - obs[:10].mean()
    std = obs.std()
    return slope, diff, std


def build_dataset(n=2000, seed=99):
    rng = np.random.default_rng(seed)
    X, AUX, y = [], [], []
    for i in range(n):
        risk = rng.uniform(0.05, 0.85)
        temps, currents, will_degrade, _ = simulate_series(rng, risk)
        obs_t = rolling_mean(temps[:OBS_WINDOW])
        obs_c = rolling_mean(currents[:OBS_WINDOW])
        # нормализация относительно собственной базовой линии юнита (первые 14 дней)
        t_mean, t_std = obs_t[:14].mean(), obs_t[:14].std() + 1e-3
        c_mean, c_std = obs_c[:14].mean(), obs_c[:14].std() + 1e-3
        day_idx = np.linspace(0, 1, OBS_WINDOW)
        feat = np.stack([(obs_t - t_mean) / t_std, (obs_c - c_mean) / c_std, day_idx], axis=1)
        X.append(feat.astype(np.float32))

        st, dt, sdt = trend_features(obs_t)
        sc, dc, sdc = trend_features(obs_c)
        AUX.append(np.array([st, dt, sdt, sc, dc, sdc], dtype=np.float32))
        y.append(1.0 if will_degrade else 0.0)
    X = np.array(X, dtype=np.float32)
    AUX = np.array(AUX, dtype=np.float32)
    # стандартизация вспомогательных признаков по всей выборке
    aux_mean, aux_std = AUX.mean(axis=0), AUX.std(axis=0) + 1e-6
    AUX = (AUX - aux_mean) / aux_std
    y = np.array(y, dtype=np.float32)
    return X, AUX, y, aux_mean, aux_std


class SeriesDataset(Dataset):
    def __init__(self, X, AUX, y):
        self.X = X
        self.AUX = AUX
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.AUX[idx]), torch.tensor(self.y[idx])


class EarlyWarningLSTM(nn.Module):
    """LSTM (учится на сырых 90-точечных рядах температуры/тока) + явные
    признаки тренда (наклон/дельта/дисперсия), объединённые в общей голове.
    Гибридная схема: нейросеть не ограничена только заранее заданными
    признаками (может доучиться на нюансах формы кривой), но получает и
    явную статистическую подсказку — так же, как реальные predictive-
    maintenance системы комбинируют feature engineering с deep learning."""

    def __init__(self, n_features=3, hidden=48, n_aux=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + n_aux, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x, aux):
        out, (h_n, _) = self.lstm(x)
        mean_pool = out.mean(dim=1)
        last = h_n[-1]
        combined = torch.cat([mean_pool, last, aux], dim=1)
        return self.head(combined).squeeze(-1)


def main():
    torch.manual_seed(0)
    X, AUX, y, aux_mean, aux_std = build_dataset(n=2400)
    n_val = 400
    X_train, AUX_train, y_train = X[n_val:], AUX[n_val:], y[n_val:]
    X_val, AUX_val, y_val = X[:n_val], AUX[:n_val], y[:n_val]

    train_loader = DataLoader(SeriesDataset(X_train, AUX_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(SeriesDataset(X_val, AUX_val, y_val), batch_size=128, shuffle=False)

    model = EarlyWarningLSTM(hidden=48)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = 0.0
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for xb, auxb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb, auxb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, auxb, yb in val_loader:
                pred = (torch.sigmoid(model(xb, auxb)) > 0.5).float()
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        best_val = max(best_val, val_acc)
        print(f"epoch {epoch+1}/30 loss={total_loss/len(X_train):.4f} val_acc={val_acc:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "obs_window": OBS_WINDOW,
        "aux_mean": aux_mean.tolist(),
        "aux_std": aux_std.tolist(),
        "best_val_acc": best_val,
    }, os.path.join(MODEL_DIR, "early_warning_lstm.pt"))
    print(f"LSTM early-warning модель сохранена. Лучшая val_acc: {best_val:.4f}")


if __name__ == "__main__":
    main()
