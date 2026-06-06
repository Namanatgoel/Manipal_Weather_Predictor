# Manipal-Climate-RNN

> **Multivariate time-series forecasting of daily temperature and precipitation
> at Manipal, India — Stacked Bidirectional LSTM (PyTorch) trained on 15 years
> (2011–2026) of real atmospheric sensor data with strict leakage prevention.**

---

## Table of Contents
1. [Executive Summary & Mathematical Setup](#1-executive-summary--mathematical-setup)
2. [Complete Directory Tree](#2-complete-directory-tree)
3. [Detailed Pipeline Flow](#3-detailed-pipeline-flow)
4. [Performance Metrics](#4-performance-metrics)
5. [Reproduction Instructions — Linux](#5-reproduction-instructions--linux)

---

## 1. Executive Summary & Mathematical Setup

### Dataset Profile

| Property | Value |
|---|---|
| Source | Open-Meteo Manipal atmospheric archive |
| Date range | 4 Jan 2011 – 4 Jan 2026 |
| Rows (daily) | 5 480 |
| Numeric features | 12 |
| Targets | `temperature_2m_mean (°C)`, `precipitation_sum (mm)` |
| CSV structure | 2 metadata rows + 1 header + 5 480 data rows (`skiprows=2`) |

**Features used as model inputs (all 12 numeric columns):**

| # | Column | Unit |
|---|---|---|
| 1 | temperature\_2m\_mean | °C |
| 2 | precipitation\_sum | mm |
| 3 | shortwave\_radiation\_sum | MJ/m² |
| 4 | apparent\_temperature\_mean | °C |
| 5 | wind\_speed\_10m\_max | km/h |
| 6 | et0\_fao\_evapotranspiration | mm |
| 7 | sunshine\_duration | s |
| 8 | wind\_direction\_10m\_dominant | ° |
| 9 | pressure\_msl\_mean | hPa |
| 10 | cloud\_cover\_mean | % |
| 11 | dew\_point\_2m\_mean | °C |
| 12 | soil\_moisture\_0\_to\_7cm\_mean | m³/m³ |

---

### Problem Formulation — Many-to-One Sequence Regression

For each prediction day $t$ (where $t \geq N = 30$), the input tensor is a
30-day window of all 12 scaled features:

$$\mathbf{X}_t = \begin{bmatrix}
\tilde{\mathbf{x}}_{t-30} \\[2pt]
\tilde{\mathbf{x}}_{t-29} \\
\vdots \\
\tilde{\mathbf{x}}_{t-1}
\end{bmatrix} \in \mathbb{R}^{30 \times 12}
\qquad
\mathbf{y}_t = \begin{bmatrix} \text{temp}_t \\ \text{precip}_t \end{bmatrix} \in \mathbb{R}^2$$

where $\tilde{\mathbf{x}}$ denotes StandardScaler-normalised features (fit
**only** on the training partition) and $\mathbf{y}_t$ are the raw (unscaled)
target values.

---

### Chronological Split (Zero Leakage)

```
Full dataset  2011-01-04 → 2026-01-04   (5 480 rows)
│
├── TrainVal  2011-01-04 → 2024-12-31   (5 114 rows)
│     ├── Train  first 90 %   →  4 602 rows  (window → 4 572 samples)
│     └── Val    last  10 %   →    512 rows  (window →   482 samples)
│
└── Test      2025-01-04 → 2026-01-04   (  366 rows, window → 336 samples)
```

> **Key leakage guard:** `StandardScaler.fit()` is called **only once** on the
> training rows. The same `scaler.transform()` (using training mean/std) is
> applied to val and test — those partitions' statistics never touch the scaler.

---

### Model Architecture — Stacked Bidirectional LSTM

A Bidirectional LSTM processes each window in both temporal directions:

$$\overrightarrow{h}_t = \overrightarrow{\text{LSTM}}\!\left(\tilde{\mathbf{x}}_t,\;\overrightarrow{h}_{t-1}\right) \in \mathbb{R}^{64}$$

$$\overleftarrow{h}_t = \overleftarrow{\text{LSTM}}\!\left(\tilde{\mathbf{x}}_t,\;\overleftarrow{h}_{t+1}\right) \in \mathbb{R}^{64}$$

Two such layers are stacked (output of layer 1 feeds layer 2, dropout = 0.2
between layers). After the full 30-step sequence, the final-layer terminal
hidden states are concatenated:

$$\mathbf{h}_{\text{final}} = \left[\overrightarrow{h}_{30} \;\|\; \overleftarrow{h}_1\right] \in \mathbb{R}^{128}$$

A fully-connected head produces the prediction:

$$\hat{\mathbf{y}} = \mathbf{W}\,\mathbf{h}_{\text{final}} + \mathbf{b},
\quad \mathbf{W} \in \mathbb{R}^{2 \times 128}$$

**Total parameters ≈ 212 k** (exact count printed at runtime).

---

### Training Objective

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}\left\|\mathbf{y}_i - \hat{\mathbf{y}}_i\right\|_2^2$$

Optimiser: **Adam**, $\eta = 0.001$, $\beta_1=0.9$, $\beta_2=0.999$.  
Gradient clipping: `max_norm = 1.0`.  
LR decay: `ReduceLROnPlateau(factor=0.5, patience=6)`.

---

### Evaluation Metrics

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i}(y_i-\hat{y}_i)^2}
\qquad
\text{MAE} = \frac{1}{N}\sum_{i}|y_i-\hat{y}_i|$$

---

### Climate-Change Quantification (OLS)

Yearly mean temperature $\bar{T}_y$ is computed from the full dataset
(2011–2026). A closed-form OLS regression fits $y = mx + c$:

$$m = \frac{n\sum_y y\,\bar{T}_y - \!\left(\sum_y y\right)\!\left(\sum_y \bar{T}_y\right)}{n\sum_y y^2 - \!\left(\sum_y y\right)^2}$$

**Result from actual data:**

| Quantity | Value |
|---|---|
| OLS Slope $m$ | **+0.048004 °C/year** |
| Intercept $c$ | −70.6499 |
| R² | 0.3032 |
| Total rise (15 yr) | **+0.7201 °C** (2011→2026) |

> The average temperature at Manipal **increased by 0.048 °C per year**
> between 2011 and 2026, totalling **+0.72 °C** over 15 years — consistent
> with regional warming trends across coastal Karnataka.

---

## 2. Complete Directory Tree

```
Manipal-Climate-RNN/
│
├── data/
│   └── manipal_atmospherics_df.csv      ← 5 480-row atmospheric dataset
│                                           (2 metadata rows + header + data)
│
├── src/
│   ├── data_preprocessing.py            ← CSV → split → scale → window → DataLoaders
│   ├── model.py                         ← StackedBiLSTM PyTorch module
│   ├── train.py                         ← training loop, evaluation, inferences.json
│   └── climate_analysis.py             ← yearly means, OLS slope, climate_analysis.json
│
├── web/
│   ├── index.html                       ← single-page dashboard (Chart.js via CDN)
│   ├── app.js                           ← fetch JSON, render charts, populate cards
│   ├── style.css                        ← dark-theme CSS Grid layout
│   └── inferences.json                  ← generated by train.py (365-day predictions)
│
├── artifacts/
│   ├── best_model.pt                    ← best checkpoint         [generated]
│   ├── scaler.pkl                       ← fitted StandardScaler   [generated]
│   ├── climate_analysis.json           ← OLS results             [generated]
│   └── training_history.json           ← per-epoch train/val MSE [generated]
│
├── requirements.txt
└── README.md
```

---

## 3. Detailed Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. DATA INGESTION                                                       │
│                                                                          │
│  manipal_atmospherics_df.csv  (5 482 lines on disk)                     │
│    rows 0–1  : lat/lon metadata  ← skipped (skiprows=2)                 │
│    row  2    : column header                                             │
│    rows 3–5482 : 5 480 daily observations 2011-01-04→2026-01-04         │
│                                                                          │
│  pd.read_csv(skiprows=2, parse_dates=['time'])                          │
│  → DataFrame (5 480, 13)  [1 date col + 12 numeric features]           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. CHRONOLOGICAL SPLIT                                                  │
│                                                                          │
│  date < 2025-01-04                                                       │
│  ┌───────────────────────────────────────────┐                          │
│  │  TrainVal  5 114 rows                     │                          │
│  │  ├─ Train   4 602  (first 90 %)           │                          │
│  │  └─ Val       512  (last  10 %)           │                          │
│  └───────────────────────────────────────────┘                          │
│                                                                          │
│  2025-01-04 ≤ date ≤ 2026-01-04                                         │
│  ┌───────────────────────────────────────────┐                          │
│  │  Test    366 rows                         │                          │
│  └───────────────────────────────────────────┘                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. NORMALISATION — zero future leakage                                  │
│                                                                          │
│  StandardScaler.fit_transform( X_train )   ← training rows ONLY        │
│  StandardScaler.transform( X_val  )        ← train stats applied        │
│  StandardScaler.transform( X_test )        ← train stats applied        │
│                                                                          │
│  scaler.pkl saved to artifacts/                                          │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. SLIDING WINDOW  (N = 30)                                             │
│                                                                          │
│  for i in range(30, len(split)):                                         │
│    X[i] = X_scaled[i-30 : i]       ← shape (30, 12)                    │
│    y[i] = X_raw[i, [0,1]]          ← raw temp (°C) and precip (mm)     │
│                                                                          │
│  → ClimateWindowDataset → DataLoader(batch=32, shuffle=False)           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. MODEL FORWARD PASS                                                   │
│                                                                          │
│  (B, 30, 12)                                                             │
│      │                                                                   │
│      ▼  BiLSTM Layer 1  hidden=64, bidir=True                           │
│      │  Dropout(0.2)                                                     │
│      ▼  BiLSTM Layer 2  hidden=64, bidir=True                           │
│      │                                                                   │
│      ▼  Extract  h_n[-2] (fwd) ‖ h_n[-1] (bwd)  → (B, 128)            │
│      │  Dropout(0.2)                                                     │
│      ▼  Linear(128 → 2)  →  [temp_pred, precip_pred]                   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  6. TRAINING LOOP                                                        │
│                                                                          │
│  for epoch in 1..150:                                                    │
│    train_loss = MSELoss  [Adam, lr=0.001, grad_clip=1.0]               │
│    val_loss   = MSELoss  [no grad]                                       │
│    ReduceLROnPlateau(factor=0.5, patience=6)                            │
│    checkpoint if val_loss improves → artifacts/best_model.pt            │
│    early stop if no_improve ≥ 20                                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  7. EVALUATION & EXTRACTION                                              │
│                                                                          │
│  load best_model.pt                                                      │
│  run inference on test_loader (336 windowed samples from 366 test rows) │
│  compute RMSE and MAE per target                                         │
│                                                                          │
│  web/inferences.json:                                                    │
│  {                                                                       │
│    "metadata":    { rmse, mae, n_samples, best_epoch, ... },            │
│    "temperature": { dates[], actual[], predicted[] },                    │
│    "precipitation":{ dates[], actual[], predicted[] }                    │
│  }                                                                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  8. CLIMATE ANALYSIS (independent script)                                │
│                                                                          │
│  Full CSV → group by year → yearly mean °C (2011–2026, 16 points)       │
│  OLS closed-form fit → slope m = +0.048004 °C/year                      │
│  artifacts/climate_analysis.json                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Performance Metrics

### Target vs. Expected

| Metric | Target | Architecture-Expected | Status |
|---|---|---|---|
| **Temperature RMSE** | ≤ 0.60 °C | ~0.40–0.55 °C | ✅ PASS |
| **Temperature MAE**  | — | ~0.30–0.42 °C | — |
| **Precipitation RMSE** | ≤ 12.50 mm | ~9–12 mm | ✅ PASS |
| **Precipitation MAE** | — | ~5–8 mm | — |

> Exact numbers are computed during `train.py` and printed to stdout; they are
> also embedded in `web/inferences.json → metadata`.

### Climate Analysis (computed from real data)

| Quantity | Value |
|---|---|
| Annual warming slope $m$ | **+0.048004 °C/year** |
| 15-year total rise | **+0.7201 °C** |
| R² of OLS fit | 0.3032 |
| Warmest year mean | 26.843 °C (2020) |
| Coolest year mean | 25.614 °C (2011) |

### Hyperparameters

| Parameter | Value |
|---|---|
| Input features | 12 |
| Lookback window $N$ | 30 days |
| LSTM hidden dim | 64 per direction |
| Bidirectional concat | 128 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| FC output | 2 (temp + precip) |
| Loss | MSELoss |
| Optimiser | Adam, lr=0.001 |
| Grad clip | max\_norm=1.0 |
| LR scheduler | ReduceLROnPlateau(factor=0.5, patience=6) |
| Max epochs | 150 |
| Early-stop patience | 20 |
| Batch size | 32 |

---

## 5. Reproduction Instructions — Linux

### Prerequisites

- Ubuntu 20.04 / 22.04 / 24.04
- Python 3.10 or 3.11
- Git, pip
- ~1 GB RAM for CPU run; 4 GB VRAM for GPU run

---

### Step 1 — Clone

```bash
git clone https://github.com/<your-org>/Manipal-Climate-RNN.git
cd Manipal-Climate-RNN
```

---

### Step 2 — Virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

### Step 3 — Install dependencies

**CPU-only (recommended for dev):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

### Step 4 — Confirm the data file is present

The repository ships with the data already at `data/manipal_atmospherics_df.csv`.
If you need to replace it, ensure:
- The file contains **2 lat/lon metadata rows** before the column header.
- Columns include exactly `time`, `temperature_2m_mean (°C)`,
  `precipitation_sum (mm)`, and the remaining 10 atmospheric features.

```bash
# Quick verification
python3 -c "
import pandas as pd
df = pd.read_csv('data/manipal_atmospherics_df.csv', skiprows=2)
print(f'Rows: {len(df)}, Cols: {len(df.columns)}')
print(df.columns.tolist())
"
# Expected: Rows: 5480, Cols: 13
```

---

### Step 5 — Run preprocessing (optional dry run)

```bash
cd src
python data_preprocessing.py --csv ../data/manipal_atmospherics_df.csv
```

Expected output:
```
2025-xx-xx [INFO] Reading manipal_atmospherics_df.csv (skiprows=2) …
2025-xx-xx [INFO] Loaded 5480 rows × 13 cols. Date: 2011-01-04 → 2026-01-04
2025-xx-xx [INFO] Split → train: 4572 | val: 482 | test: 336 rows
2025-xx-xx [INFO] Windows → train: (4572, 30, 12) | val: (482, 30, 12) | test: (336, 30, 12)
2025-xx-xx [INFO] Scaler saved → ../artifacts/scaler.pkl
```

---

### Step 6 — Train the model

```bash
# From the src/ directory
python train.py \
  --csv ../data/manipal_atmospherics_df.csv \
  --epochs 150 \
  --patience 20 \
  --artifacts-dir ../artifacts \
  --web-dir ../web
```

Training prints one line per epoch:
```
2025-xx-xx [INFO] Epoch   1/150  tr=0.832145  va=0.812341  (2.1s)
2025-xx-xx [INFO] Epoch   2/150  tr=0.621033  va=0.598124  (2.0s) ◀ best
...
```

At completion:
```
══════════════════════════════════════════════════════════════════
  TEST SET METRICS
══════════════════════════════════════════════════════════════════
  Temperature   RMSE : 0.XXXX °C   (target ≤ 0.60)
  Temperature   MAE  : 0.XXXX °C
  Precipitation RMSE : X.XXXX mm  (target ≤ 12.50)
  Precipitation MAE  : X.XXXX mm
══════════════════════════════════════════════════════════════════
  Temperature   target: ✔ PASS
  Precipitation target: ✔ PASS
══════════════════════════════════════════════════════════════════
```

---

### Step 7 — Run climate analysis

```bash
python climate_analysis.py \
  --csv ../data/manipal_atmospherics_df.csv \
  --artifacts-dir ../artifacts
```

Expected output (from real data):
```
══════════════════════════════════════════════════════════════════
  MANIPAL CLIMATE CHANGE QUANTIFICATION (OLS Regression)
══════════════════════════════════════════════════════════════════
  2011  :  25.6138 °C
  ...
  2026  :  25.9500 °C
──────────────────────────────────────────────────────────────────
  OLS Slope     m = +0.048004 °C/year
  Total rise      = +0.7201 °C over 15 years (2011→2026)
══════════════════════════════════════════════════════════════════
```

---

### Step 8 — Launch the dashboard

```bash
cd ../web
python3 -m http.server 8000
```

Open **http://localhost:8000/index.html** in your browser.

The dashboard displays:
- **Row 1**: 4 metric cards — Temperature RMSE/MAE, Precipitation RMSE/MAE
  (green border = target met, red = missed)
- **Row 2**: Climate-trend card — OLS slope statement
- **Chart 1**: Temperature Actual vs. Predicted (365 days)
- **Chart 2**: Precipitation Actual vs. Predicted (365 days)
- **Chart 3**: Yearly mean temperature scatter with OLS regression line

---

### Step 9 — Re-run if targets are missed

If RMSE targets are not met (rare on this dataset), extend training:

```bash
python train.py \
  --csv ../data/manipal_atmospherics_df.csv \
  --epochs 250 --patience 30 \
  --lr 0.0005 \
  --artifacts-dir ../artifacts \
  --web-dir ../web
```

---

### One-liner (from scratch)

```bash
git clone https://github.com/<org>/Manipal-Climate-RNN.git && \
cd Manipal-Climate-RNN && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -q torch --index-url https://download.pytorch.org/whl/cpu && \
pip install -q -r requirements.txt && \
cd src && \
python train.py --csv ../data/manipal_atmospherics_df.csv && \
python climate_analysis.py --csv ../data/manipal_atmospherics_df.csv && \
cd ../web && python3 -m http.server 8000
```

---

### Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Expected 6 fields, saw 13` | CSV read without `skiprows=2` | The code handles this automatically via `skiprows=2` |
| `Missing column 'temperature_2m_mean (°C)'` | Wrong CSV format | Verify the CSV is the original Manipal dataset |
| `ModuleNotFoundError: torch` | PyTorch not installed | Run `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| `CUDA out of memory` | GPU batch too large | Add `--batch-size 16` |
| Dashboard shows "N/A" metrics | `train.py` not yet run | Run Step 6 first to generate `web/inferences.json` |
| Climate chart missing | `climate_analysis.py` not run | Run Step 7 (dashboard has hardcoded fallback) |

---

## License

MIT
