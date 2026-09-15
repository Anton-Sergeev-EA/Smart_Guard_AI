"""
STARGUARD AI — инференс LSTM-модели раннего предупреждения.

Принимает первые OBS_WINDOW дней телеметрии юнита (температура, ток) и
возвращает вероятность будущей деградации — до того, как классический
пороговый детектор (C++ telemetry_engine) вообще увидит повод для тревоги.
"""
import os
import numpy as np
import torch

from train_predictive_model import EarlyWarningLSTM, rolling_mean, trend_features, OBS_WINDOW

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "data", "models", "early_warning_lstm.pt")

_cache = {}


def _load():
    if "model" in _cache:
        return _cache["model"], _cache["aux_mean"], _cache["aux_std"]
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = EarlyWarningLSTM(hidden=48)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    aux_mean = np.array(ckpt["aux_mean"], dtype=np.float32)
    aux_std = np.array(ckpt["aux_std"], dtype=np.float32)
    _cache.update(model=model, aux_mean=aux_mean, aux_std=aux_std)
    return model, aux_mean, aux_std


def predict_early_warning(temps, currents):
    """temps/currents — списки минимум OBS_WINDOW дней. Возвращает вероятность
    будущей деградации (0-1) на основе только первых OBS_WINDOW точек."""
    model, aux_mean, aux_std = _load()
    temps = np.asarray(temps[:OBS_WINDOW], dtype=np.float64)
    currents = np.asarray(currents[:OBS_WINDOW], dtype=np.float64)
    if len(temps) < OBS_WINDOW:
        return None

    obs_t = rolling_mean(temps)
    obs_c = rolling_mean(currents)
    t_mean, t_std = obs_t[:14].mean(), obs_t[:14].std() + 1e-3
    c_mean, c_std = obs_c[:14].mean(), obs_c[:14].std() + 1e-3
    day_idx = np.linspace(0, 1, OBS_WINDOW)
    seq = np.stack([(obs_t - t_mean) / t_std, (obs_c - c_mean) / c_std, day_idx], axis=1).astype(np.float32)

    st, dt, sdt = trend_features(obs_t)
    sc, dc, sdc = trend_features(obs_c)
    aux = np.array([st, dt, sdt, sc, dc, sdc], dtype=np.float32)
    aux = (aux - aux_mean) / aux_std

    with torch.no_grad():
        x = torch.from_numpy(seq).unsqueeze(0)
        a = torch.from_numpy(aux).unsqueeze(0)
        prob = torch.sigmoid(model(x, a)).item()
    return round(float(prob), 4)


if __name__ == "__main__":
    import json
    from train_predictive_model import simulate_series
    rng = np.random.default_rng(1)
    t, c, will_degrade, start = simulate_series(rng, 0.7)
    p = predict_early_warning(t, c)
    print(json.dumps({"will_degrade_actual": will_degrade, "degrade_start_day": start, "predicted_prob": p}, ensure_ascii=False))
