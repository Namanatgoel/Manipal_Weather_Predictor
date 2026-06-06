"""
train.py — Manipal-Climate-RNN
═══════════════════════════════════════════════════════════════════════════════
Full training, evaluation, and inference-export pipeline.

Spec
  Loss      : MSELoss
  Optimiser : Adam  lr=0.001
  Target    : Temperature RMSE ≤ 0.6 °C  |  Precipitation RMSE ≤ 12.5 mm
  Output    : web/inferences.json  (365-day actual vs predicted)
"""

import json, logging, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local imports (run from project root or src/)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_and_preprocess
from model import StackedBiLSTM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────
CSV_PATH      = "data/manipal_atmospherics_df.csv"
ARTIFACTS_DIR = Path("artifacts")
WEB_DIR       = Path("web")
LOOKBACK      = 30
BATCH_SIZE    = 32
HIDDEN_DIM    = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
LR            = 0.001
EPOCHS        = 150
PATIENCE      = 20


# ── Metric helpers ─────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Single epoch ───────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = criterion(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


# ── Full inference (no grad) ───────────────────────────────────────────────────
def predict(model, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.append(model(Xb.to(device)).cpu().numpy())
            tgts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(tgts)


# ── Main ───────────────────────────────────────────────────────────────────────
def train(
    csv_path      = CSV_PATH,
    lookback      = LOOKBACK,
    batch_size    = BATCH_SIZE,
    hidden_dim    = HIDDEN_DIM,
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
    lr            = LR,
    epochs        = EPOCHS,
    patience      = PATIENCE,
    artifacts_dir = ARTIFACTS_DIR,
    web_dir       = WEB_DIR,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(web_dir).mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    tr_l, va_l, te_l, scaler, meta = load_and_preprocess(
        csv_path=csv_path, lookback=lookback,
        batch_size=batch_size, artifacts_dir=str(artifacts_dir))
    n_feat     = meta["n_features"]     # 12
    test_dates = meta["test_dates"]

    # ── Model ────────────────────────────────────────────────────────────────
    model = StackedBiLSTM(
        input_dim=n_feat, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout, output_dim=2
    ).to(device)
    log.info("%s", model)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=6)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val, best_epoch, no_improve = float("inf"), 0, 0
    history = {"train_loss": [], "val_loss": []}
    ckpt    = Path(artifacts_dir) / "best_model.pt"

    for epoch in range(1, epochs + 1):
        t0       = time.time()
        tr_loss  = run_epoch(model, tr_l, criterion, optimizer, device, True)
        va_loss  = run_epoch(model, va_l, criterion, optimizer, device, False)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        mark = ""
        if va_loss < best_val:
            best_val, best_epoch, no_improve = va_loss, epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": va_loss, "meta": meta}, ckpt)
            mark = " ◀ best"
        else:
            no_improve += 1

        log.info("Epoch %3d/%d  tr=%.6f  va=%.6f  (%.1fs)%s",
                 epoch, epochs, tr_loss, va_loss, time.time()-t0, mark)

        if no_improve >= patience:
            log.info("Early stop at epoch %d (best=%d)", epoch, best_epoch)
            break

    # ── Evaluate ─────────────────────────────────────────────────────────────
    log.info("Loading best checkpoint (epoch %d) …", best_epoch)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])

    preds, tgts = predict(model, te_l, device)

    temp_pred,   temp_true   = preds[:, 0], tgts[:, 0]
    precip_pred, precip_true = preds[:, 1], tgts[:, 1]

    t_rmse = rmse(temp_true,   temp_pred)
    t_mae  = mae(temp_true,    temp_pred)
    p_rmse = rmse(precip_true, precip_pred)
    p_mae  = mae(precip_true,  precip_pred)

    divider = "=" * 62
    print(f"\n{divider}")
    print("  TEST SET METRICS")
    print(divider)
    print(f"  Temperature   RMSE : {t_rmse:.4f} °C   (target ≤ 0.60)")
    print(f"  Temperature   MAE  : {t_mae:.4f} °C")
    print(f"  Precipitation RMSE : {p_rmse:.4f} mm  (target ≤ 12.50)")
    print(f"  Precipitation MAE  : {p_mae:.4f} mm")
    print(divider)
    print(f"  Temperature   target: {'✔ PASS' if t_rmse <= 0.6  else '✘ MISS'}")
    print(f"  Precipitation target: {'✔ PASS' if p_rmse <= 12.5 else '✘ MISS'}")
    print(divider + "\n")

    # ── inferences.json ──────────────────────────────────────────────────────
    n = len(temp_pred)
    dates_aligned = (test_dates[:n] if test_dates and len(test_dates) >= n
                     else [str(i) for i in range(n)])

    inferences = {
        "metadata": {
            "test_start":          "2025-01-04",
            "test_end":            "2026-01-04",
            "n_samples":           n,
            "lookback":            lookback,
            "best_epoch":          best_epoch,
            "best_val_loss":       round(float(best_val), 6),
            "temperature_rmse":    round(t_rmse, 4),
            "temperature_mae":     round(t_mae,  4),
            "precipitation_rmse":  round(p_rmse, 4),
            "precipitation_mae":   round(p_mae,  4),
        },
        "temperature": {
            "dates":     dates_aligned,
            "actual":    [round(float(v), 4) for v in temp_true],
            "predicted": [round(float(v), 4) for v in temp_pred],
        },
        "precipitation": {
            "dates":     dates_aligned,
            "actual":    [round(float(v), 4) for v in precip_true],
            "predicted": [round(float(v), 4) for v in precip_pred],
        },
    }

    inf_path = Path(web_dir) / "inferences.json"
    with open(inf_path, "w") as fh:
        json.dump(inferences, fh, indent=2)
    log.info("Inferences saved → %s (%d records)", inf_path, n)

    # persist history
    with open(Path(artifacts_dir) / "training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    return {"t_rmse": t_rmse, "t_mae": t_mae,
            "p_rmse": p_rmse, "p_mae": p_mae,
            "best_epoch": best_epoch}


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train Manipal-Climate-RNN")
    ap.add_argument("--csv",           default=CSV_PATH)
    ap.add_argument("--lookback",      type=int,   default=LOOKBACK)
    ap.add_argument("--batch-size",    type=int,   default=BATCH_SIZE)
    ap.add_argument("--hidden-dim",    type=int,   default=HIDDEN_DIM)
    ap.add_argument("--num-layers",    type=int,   default=NUM_LAYERS)
    ap.add_argument("--dropout",       type=float, default=DROPOUT)
    ap.add_argument("--lr",            type=float, default=LR)
    ap.add_argument("--epochs",        type=int,   default=EPOCHS)
    ap.add_argument("--patience",      type=int,   default=PATIENCE)
    ap.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    ap.add_argument("--web-dir",       default=str(WEB_DIR))
    a = ap.parse_args()

    results = train(
        csv_path=a.csv, lookback=a.lookback, batch_size=a.batch_size,
        hidden_dim=a.hidden_dim, num_layers=a.num_layers, dropout=a.dropout,
        lr=a.lr, epochs=a.epochs, patience=a.patience,
        artifacts_dir=Path(a.artifacts_dir), web_dir=Path(a.web_dir),
    )
    sys.exit(0)
