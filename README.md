
# BACalc (Qt Version)

BACalc Biological Age Program wirrten in Python Qt:
**PCA > BAS > T-scale > Dubina-corrected BAc**

## Features
- Load/merge multiple CSVs
- Choose age column and an optional binary split column
- Select biomarker columns (multi-select, use CTRL+Click)
- Optional MAP computation from `Systolic_BP` and `Diastolic_BP`
- Runs the full pipeline:
  - Z-score > PCA (PC1) > BAS
  - T-scale to chronological age (CA)
  - Dubina correction to remove CA correlation
- Saves: `ba_predictions.csv`, `pca_loadings.csv`, `ba_coefficients.csv`, `ba_equations.txt`
- Generates scatter plots: Age vs BA, Age vs BAc (global + per-group)

## Quick start
```bash
# 1) Create venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
python app.py
```

The BACalc writes outputs to the chosen "Output directory" (default: `./output_results`).

## Windows Exe Release
Users running Windows 64 operating system can download our exe builds for a simple and easy run.
