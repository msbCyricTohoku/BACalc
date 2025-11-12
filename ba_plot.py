from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

# import matplotlib
import matplotlib.pyplot as plt


def sanitize_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def plot_ba_results(pred_csv: Path, out_dir: Path, log=print):
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    if not {"Age", "BA", "BAc"}.issubset(df.columns):
        raise ValueError("Predictions file lacks needed columns (Age, BA, BAc).")

    def plot_xy(x, y, title, fname):
        plt.figure()
        groups = df["Group"].unique() if "Group" in df.columns else ["All"]
        for g in groups:
            sub = df[df["Group"] == g] if "Group" in df.columns else df
            plt.scatter(sub[x], sub[y], label=str(g), alpha=0.6, s=14)
        # identity line
        mn = float(np.nanmin(df[x].to_numpy()))
        mx = float(np.nanmax(df[x].to_numpy()))
        lo, hi = mn, mx
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
        if len(groups) > 1:
            plt.legend()
        plt.tight_layout()
        outp = plots_dir / fname
        plt.savefig(outp, dpi=150)
        plt.close()
        log(f"Saved plot: {outp}")
        return outp

    ba_all = plot_xy("Age", "BA", "BA vs Age (All Groups)", "plot_ba_vs_age.png")
    bac_all = plot_xy("Age", "BAc", "BAc vs Age (All Groups)", "plot_bac_vs_age.png")

    group_plots = []
    if "Group" in df.columns:
        for g in df["Group"].unique():
            # sub = df[df["Group"] == g]
            tag = sanitize_for_filename(str(g))
            group_plots.append(
                plot_xy("Age", "BA", f"BA vs Age — {g}", f"plot_ba_vs_age_{tag}.png")
            )
            group_plots.append(
                plot_xy("Age", "BAc", f"BAc vs Age — {g}", f"plot_bac_vs_age_{tag}.png")
            )

    return {
        "ba_all": ba_all,
        "bac_all": bac_all,
        "group_plots": group_plots,
        "plots_dir": plots_dir,
    }
