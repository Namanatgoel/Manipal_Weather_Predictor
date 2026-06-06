"""
data_preprocessing.py — Manipal-Climate-RNN
═══════════════════════════════════════════════════════════════════════════════
Reads the actual manipal_atmospherics_df.csv (requires skiprows=2 to skip the
lat/lon metadata header), applies a strict chronological split, fits a
StandardScaler exclusively on training data, and returns sliding-window
PyTorch DataLoaders.

CSV layout (after skiprows=2)
  col 0  : time                              — date string
  cols 1–12: 12 numeric atmospheric features

Split boundaries (strict chronological, zero leakage)
  TrainVal : 2011-01-04 – 2024-12-31   (5 114 rows)
    └─ Train : first 90 % of TrainVal  (4 602 rows)
    └─ Val   : last  10 % of TrainVal  (  512 rows)
  Test     : 2025-01-04 – 2026-01-04  (  366 rows)

Sliding window (N = 30)
  X[i] = scaled_features[i-30 : i]   shape (30, 12)
  y[i] = raw_targets[i]               shape (2,)   [temp_degC, precip_mm]
"""

import os, pickle, logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CSV_SKIPROWS = 2          # skip the lat/lon metadata lines
DATE_COL     = "time"
TARGET_COLS  = ["temperature_2m_mean (°C)", "precipitation_sum (mm)"]
TEST_START   = pd.Timestamp("2025-01-04")
TEST_END     = pd.Timestamp("2026-01-04")
LOOKBACK     = 30
BATCH_SIZE   = 32

FEATURE_COLS = [
    "temperature_2m_mean (°C)",
    "precipitation_sum (mm)",
    "shortwave_radiation_sum (MJ/m²)",
    "apparent_temperature_mean (°C)",
    "wind_speed_10m_max (km/h)",
    "et0_fao_evapotranspiration (mm)",
    "sunshine_duration (s)",
    "wind_direction_10m_dominant (°)",
    "pressure_msl_mean (hPa)",
    "cloud_cover_mean (%)",
    "dew_point_2m_mean (°C)",
    "soil_moisture_0_to_7cm_mean (m³/m³)",
]
N_FEATURES   = len(FEATURE_COLS)   # 12
TARGET_IDX   = [FEATURE_COLS.index(c) for c in TARGET_COLS]  # [0, 1]


# ── Dataset ────────────────────────────────────────────────────────────────────
class ClimateWindowDataset(Dataset):
    """
    Wraps pre-built (X_scaled, y_raw) numpy arrays as a PyTorch Dataset.

    X shape : (N, LOOKBACK, N_FEATURES)  — StandardScaler-normalised inputs
    y shape : (N, 2)                     — raw °C and mm targets
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, dates: list = None):
        self.X     = torch.tensor(X, dtype=torch.float32)
        self.y     = torch.tensor(y, dtype=torch.float32)
        self.dates = dates  # list[str] aligned with y rows

    def __len__(self):  return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.y[i]


# ── Window builder ─────────────────────────────────────────────────────────────
def _build_windows(X_scaled: np.ndarray,
                   X_raw:    np.ndarray,
                   target_idx: list,
                   lookback: int):
    Xs, ys = [], []
    for i in range(lookback, len(X_scaled)):
        Xs.append(X_scaled[i - lookback : i])   # (lookback, F)
        ys.append(X_raw[i, target_idx])          # (2,)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── Public API ─────────────────────────────────────────────────────────────────
def load_and_preprocess(
    csv_path:      str | Path = "data/manipal_atmospherics_df.csv",
    lookback:      int        = LOOKBACK,
    batch_size:    int        = BATCH_SIZE,
    artifacts_dir: str | Path = "artifacts",
):
    """
    Full pipeline: CSV → split → scale → window → DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    scaler                                : fitted StandardScaler (12 features)
    meta                                  : dict with sizes, dates, column info
    """
    csv_path      = Path(csv_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    log.info("Reading %s (skiprows=%d) …", csv_path.name, CSV_SKIPROWS)
    df = pd.read_csv(csv_path, skiprows=CSV_SKIPROWS, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    log.info("Loaded %d rows × %d cols. Date: %s → %s",
             len(df), len(df.columns),
             df[DATE_COL].iloc[0].date(), df[DATE_COL].iloc[-1].date())

    # Validate columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # ── 2. Chronological split ─────────────────────────────────────────────────
    mask_tv   = df[DATE_COL] < TEST_START
    mask_test = (df[DATE_COL] >= TEST_START) & (df[DATE_COL] <= TEST_END)

    tv_df   = df.loc[mask_tv,   FEATURE_COLS].reset_index(drop=True)
    test_df = df.loc[mask_test, FEATURE_COLS].reset_index(drop=True)
    test_dates = df.loc[mask_test, DATE_COL].dt.strftime("%Y-%m-%d").tolist()

    tv_split = int(len(tv_df) * 0.90)
    train_df = tv_df.iloc[:tv_split]
    val_df   = tv_df.iloc[tv_split:]

    log.info("Split → train: %d | val: %d | test: %d rows",
             len(train_df), len(val_df), len(test_df))

    # ── 3. Scale — fit ONLY on train ───────────────────────────────────────────
    X_train_raw = train_df.values.astype(np.float64)
    X_val_raw   = val_df.values.astype(np.float64)
    X_test_raw  = test_df.values.astype(np.float64)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_raw)
    X_va_s = scaler.transform(X_val_raw)
    X_te_s = scaler.transform(X_test_raw)

    log.info("Scaler fitted. mean[:3]=%s std[:3]=%s",
             np.round(scaler.mean_[:3], 3), np.round(scaler.scale_[:3], 3))

    # ── 4. Sliding window ──────────────────────────────────────────────────────
    X_tr, y_tr = _build_windows(X_tr_s, X_train_raw, TARGET_IDX, lookback)
    X_va, y_va = _build_windows(X_va_s, X_val_raw,   TARGET_IDX, lookback)
    X_te, y_te = _build_windows(X_te_s, X_test_raw,  TARGET_IDX, lookback)

    log.info("Windows → train: %s | val: %s | test: %s", X_tr.shape, X_va.shape, X_te.shape)

    # Align test_dates with windowed samples
    test_dates_w = test_dates[lookback:] if len(test_dates) > lookback else test_dates

    # ── 5. DataLoaders ────────────────────────────────────────────────────────
    train_ds = ClimateWindowDataset(X_tr, y_tr)
    val_ds   = ClimateWindowDataset(X_va, y_va)
    test_ds  = ClimateWindowDataset(X_te, y_te, dates=test_dates_w)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    # ── 6. Persist ────────────────────────────────────────────────────────────
    scaler_path = artifacts_dir / "scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    log.info("Scaler saved → %s", scaler_path)

    meta = {
        "feature_cols":   FEATURE_COLS,
        "target_cols":    TARGET_COLS,
        "target_indices": TARGET_IDX,
        "n_features":     N_FEATURES,
        "lookback":       lookback,
        "train_size":     len(X_tr),
        "val_size":       len(X_va),
        "test_size":      len(X_te),
        "test_dates":     test_dates_w,
        "scaler_path":    str(scaler_path),
    }
    return train_loader, val_loader, test_loader, scaler, meta


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys, json
    p = argparse.ArgumentParser()
    p.add_argument("--csv",           default="data/manipal_atmospherics_df.csv")
    p.add_argument("--lookback",      type=int, default=LOOKBACK)
    p.add_argument("--batch-size",    type=int, default=BATCH_SIZE)
    p.add_argument("--artifacts-dir", default="artifacts")
    a = p.parse_args()

    tr, va, te, sc, meta = load_and_preprocess(
        a.csv, a.lookback, a.batch_size, a.artifacts_dir)

    print("\n=== Preprocessing complete ===")
    for k, v in meta.items():
        if k != "test_dates":
            print(f"  {k}: {v}")
    print(f"  test_dates[0]: {meta['test_dates'][0] if meta['test_dates'] else 'N/A'}")
    sys.exit(0)
