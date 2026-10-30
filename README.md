
# BACalc (Qt Version)

BACalc Biological Age Program wirrten in Python Qt:
**PCA > BAS > T-scale > Dubina-corrected BAc and KDM Estimation**

## Screenshot
![App Screenshot](screen.png)

## Features
- Load/merge multiple CSVs
- Choose age column and an optional binary split column
- Select biomarker columns (multi-select, use CTRL+Click)
- Optional MAP computation from `Systolic_BP` and `Diastolic_BP`
- Runs two full pipeline:
  - Method 1: Runs PCA-Dubina.
  - Method 2: Runs KDM.
- Saves: `ba_predictions.csv`, `pca_loadings.csv`, `ba_coefficients.csv`, `ba_equations.txt`
- Generates global + per-group plots

## Quick start
```bash
# 1) Create venv (recommended)
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
python app.py
```

The BACalc writes outputs to the chosen "Output directory" (default: `./output_results`).

## Bug Fixes (KDM Implementation)
- **Unsigned correlation (`|r|`) in r<sub>char</sub>:** Use `abs(r)` instead of signed *r* in the denominator so negatively correlated biomarkers do not reduce it, eliminating a ~2-year bias to match R's `BioAge` package.
- **Dynamic variance floor (s²<sub>BA</sub>):** Fall back to empirical chronological age variance (`var_ca_approx`) instead of a hardcoded `0.1` clamp when variance drops below zero, preventing variance collapse and restoring Cox model convergence.
- **Reference agreement:** Biological Age estimates now closely align with reference implementations.
