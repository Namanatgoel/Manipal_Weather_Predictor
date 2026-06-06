"""
climate_analysis.py — Manipal-Climate-RNN
═══════════════════════════════════════════════════════════════════════════════
Quantifies the long-term thermal shift at Manipal using the full 15-year
timeline (2011-01-04 to 2026-01-04).

Method
  1. Load the full CSV (skiprows=2, all 5 480 rows).
  2. Group daily temperature by calendar year → yearly mean vector.
  3. Fit OLS linear regression  y = m·x + c  via closed-form normal equations.
  4. Report slope m (°C/year), intercept c, R², and total rise (2011→2026).
  5. Serialise to artifacts/climate_analysis.json.

OLS closed form
  m = [n·Σ(xy) − Σx·Σy] / [n·Σ(x²) − (Σx)²]
  c = [Σy − m·Σx] / n
"""

import json, logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

CSV_SKIPROWS = 2
DATE_COL     = "time"
TEMP_COL     = "temperature_2m_mean (°C)"
ARTIFACTS_DIR = Path("artifacts")


def _ols(x: np.ndarray, y: np.ndarray):
    """Closed-form OLS: returns (slope m, intercept c, R²)."""
    n      = len(x)
    sx, sy = x.sum(), y.sum()
    sxy    = (x * y).sum()
    sx2    = (x ** 2).sum()
    denom  = n * sx2 - sx ** 2
    if abs(denom) < 1e-12:
        raise ValueError("Degenerate x array — cannot compute OLS.")
    m  = (n * sxy - sx * sy) / denom
    c  = (sy - m * sx) / n
    yh = m * x + c
    ss_res = ((y - yh) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(m), float(c), float(r2)


def run_climate_analysis(
    csv_path:      str | Path = "data/manipal_atmospherics_df.csv",
    artifacts_dir: str | Path = ARTIFACTS_DIR,
) -> dict:
    """
    Compute OLS thermal-trend slope and save results to JSON.

    Returns dict with keys:
      slope_m, intercept_c, r2, total_rise_degC,
      years, yearly_means, regression_line, interpretation
    """
    csv_path      = Path(csv_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info("Loading %s …", csv_path.name)
    df = pd.read_csv(csv_path, skiprows=CSV_SKIPROWS, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    log.info("Loaded %d rows, date: %s → %s",
             len(df), df[DATE_COL].iloc[0].date(), df[DATE_COL].iloc[-1].date())

    if TEMP_COL not in df.columns:
        raise ValueError(f"Column '{TEMP_COL}' not found. Columns: {df.columns.tolist()}")

    # ── 2. Yearly means (2011–2026) ───────────────────────────────────────────
    df["year"] = df[DATE_COL].dt.year
    yearly = (
        df[(df["year"] >= 2011) & (df["year"] <= 2026)]
        .groupby("year")[TEMP_COL]
        .mean()
        .reset_index()
        .rename(columns={TEMP_COL: "mean_temp"})
        .sort_values("year")
    )
    log.info("Yearly means computed for %d years: %s",
             len(yearly), yearly["year"].tolist())

    years = yearly["year"].values.astype(float)
    means = yearly["mean_temp"].values

    # ── 3. OLS ────────────────────────────────────────────────────────────────
    m, c, r2 = _ols(years, means)
    total_rise = m * (years[-1] - years[0])

    # ── 4. Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  MANIPAL CLIMATE CHANGE QUANTIFICATION (OLS Regression)")
    print("=" * 62)
    for yr, mn in zip(yearly["year"].tolist(), means):
        print(f"    {yr}  :  {mn:.4f} °C")
    print("-" * 62)
    print(f"  OLS Slope     m = {m:+.6f} °C/year")
    print(f"  Intercept     c = {c:.4f}")
    print(f"  R²              = {r2:.4f}")
    print(f"  Total rise      = {total_rise:+.4f} °C over "
          f"{int(years[-1]-years[0])} years (2011→2026)")
    trend = "warming" if m > 0 else "cooling"
    print(f"  Trend           : {abs(m):.4f} °C/year {trend}")
    print("=" * 62 + "\n")

    # ── 5. Persist ────────────────────────────────────────────────────────────
    result = {
        "slope_m":       round(m,  6),
        "intercept_c":   round(c,  4),
        "r2":            round(r2, 4),
        "total_rise_degC": round(total_rise, 4),
        "years":         yearly["year"].tolist(),
        "yearly_means":  [round(float(v), 4) for v in means],
        "regression_line": {
            str(int(yr)): round(float(m * yr + c), 4)
            for yr in yearly["year"].tolist()
        },
        "interpretation": (
            f"The annual mean temperature at Manipal "
            f"{'increased' if m > 0 else 'decreased'} by "
            f"{abs(m):.4f} °C/year between 2011 and 2026, "
            f"totalling {abs(total_rise):.4f} °C over 15 years "
            f"(OLS slope m = {m:+.6f} °C/year, R² = {r2:.4f})."
        ),
    }

    out = artifacts_dir / "climate_analysis.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("Climate analysis saved → %s", out)

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",           default="data/manipal_atmospherics_df.csv")
    ap.add_argument("--artifacts-dir", default="artifacts")
    a  = ap.parse_args()
    r  = run_climate_analysis(a.csv, a.artifacts_dir)
    print(r["interpretation"])
