"""Train the LSTM or Transformer forecaster and compare it with naive baselines.

Baselines:
  persistence     next bins = last observed bin
  moving average  next bins = mean of the last 4 bins

Writes models/forecaster.pt and results/forecast_metrics.json, plus
results/forecast_examples.json (predicted vs actual) for dashboard visualization.

Usage:
    python -m forecast.train_forecaster --epochs 40 --model lstm
    python -m forecast.train_forecaster --epochs 40 --model transformer
"""
import argparse
import json

import numpy as np
import torch

from config import (FORECAST_D_MODEL, FORECAST_MODEL, FORECAST_NHEAD,
                    FORECAST_NUM_LAYERS, MAX_PHASES, RESULTS,
                    TRANSFORMER_FORECAST_MODEL)
from forecast.model import (COUNT_SCALE, LSTMForecaster, TransformerForecaster,
                            make_windows, normalise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--model", choices=["lstm", "transformer"], default="lstm",
                    help="Forecaster neural architecture")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    series_path = RESULTS / "forecast_series.npy"
    if not series_path.exists():
        raise FileNotFoundError(f"{series_path} not found. Run 'python -m forecast.collect_data' first.")

    series = np.load(series_path)
    idx = rng.permutation(len(series))
    n_val = max(1, len(series) // 5)
    val_s, train_s = series[idx[:n_val]], series[idx[n_val:]]

    def windows(ss):
        xs, ys = zip(*(make_windows(s) for s in ss))
        return np.concatenate(xs), np.concatenate(ys)

    Xtr, Ytr = windows(train_s)
    Xva, Yva = windows(val_s)
    xtr, ytr = torch.from_numpy(normalise(Xtr)), torch.from_numpy(Ytr / COUNT_SCALE)
    xva = torch.from_numpy(normalise(Xva))

    if a.model == "transformer":
        kwargs = {
            "d_model": FORECAST_D_MODEL,
            "nhead": FORECAST_NHEAD,
            "num_layers": FORECAST_NUM_LAYERS
        }
        model = TransformerForecaster(**kwargs)
        save_path = TRANSFORMER_FORECAST_MODEL if a.model == "transformer" else FORECAST_MODEL
    else:
        kwargs = {"hidden": a.hidden}
        model = LSTMForecaster(**kwargs)
        save_path = FORECAST_MODEL

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = torch.nn.SmoothL1Loss()
    history = []
    best_val_mae = float("inf")

    print(f"Training {a.model.upper()} Forecaster for {a.epochs} epochs...", flush=True)

    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = loss_fn(model(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)

        model.eval()
        with torch.no_grad():
            pred = model(xva).numpy() * COUNT_SCALE
        mae = float(np.abs(np.clip(pred, 0, None) - Yva).mean())
        history.append({"epoch": ep + 1, "train_loss": tot / len(xtr), "val_mae": mae})

        if mae < best_val_mae:
            best_val_mae = mae
            # Save checkpoint
            torch.save({
                "state_dict": model.state_dict(),
                "kwargs": kwargs,
                "arch": a.model
            }, save_path)
            # Also save to main FORECAST_MODEL path for immediate environment use
            torch.save({
                "state_dict": model.state_dict(),
                "kwargs": kwargs,
                "arch": a.model
            }, FORECAST_MODEL)

        if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
            print(f"[{a.model.upper()}] Epoch {ep+1:2d}: Train Loss {tot/len(xtr):.4f} | Val MAE {mae:.3f} veh/bin", flush=True)

    with torch.no_grad():
        pred = np.clip(model(xva).numpy() * COUNT_SCALE, 0, None)
    last = Xva[:, -1:, :MAX_PHASES].repeat(Yva.shape[1], axis=1)
    mov = Xva[:, -4:, :MAX_PHASES].mean(axis=1, keepdims=True).repeat(Yva.shape[1], axis=1)

    metrics = {
        "model_architecture": a.model,
        "lstm_mae": float(np.abs(pred - Yva).mean()),
        "persistence_mae": float(np.abs(last - Yva).mean()),
        "moving_average_mae": float(np.abs(mov - Yva).mean()),
        "lstm_rmse": float(np.sqrt(((pred - Yva) ** 2).mean())),
        "persistence_rmse": float(np.sqrt(((last - Yva) ** 2).mean())),
        "unit": "vehicles per 30-second bin per phase",
        "train_windows": int(len(Xtr)),
        "val_windows": int(len(Xva)),
        "history": history,
    }

    (RESULTS / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))

    # One validation series for plotting
    Xs, Ys = make_windows(val_s[0])
    with torch.no_grad():
        ps = np.clip(model(torch.from_numpy(normalise(Xs))).numpy() * COUNT_SCALE, 0, None)
    ex = {
        "actual": Ys[:, 0, :].tolist(),
        "lstm": ps[:, 0, :].tolist(),
        "persistence": Xs[:, -1, :MAX_PHASES].tolist()
    }
    (RESULTS / "forecast_examples.json").write_text(json.dumps(ex))
    print("\nTraining complete. Final metrics summary:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
