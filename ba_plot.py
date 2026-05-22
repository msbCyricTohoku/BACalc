from __future__ import annotations

# Developed by Mehrdad S. Beni, Haipeng Liu, Gary Tse at HKMU, Hong Kong - 2025
# Distributed under GPL V3 license
# Refined for enhanced aesthetics and spacing - 2026

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import r2_score

# Set a clean, professional aesthetic
sns.set_theme(style="whitegrid", palette="muted")

def sanitize_for_filename(s: str) -> str:
    """for filename sanitization"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def force_square_axes(ax, data_x, data_y):
    """Forces X and Y axes to share limits/ticks for 1:1 aspect ratio."""
    all_data = np.concatenate([data_x, data_y])
    all_data = all_data[~np.isnan(all_data)]

    if len(all_data) == 0:
        ax.set_aspect("equal", adjustable="box")
        return

    d_min, d_max = all_data.min(), all_data.max()
    pad = (d_max - d_min) * 0.1  # Increased padding for clarity
    lim_min, lim_max = d_min - pad, d_max + pad

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")

    locator = ticker.MaxNLocator(nbins=6)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)

def add_regression_stats(ax, x, y, line_color="#2c3e50", fill_color="#34495e"):
    """Adds regression line, CI, and R2/r stats with cleaner styling."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 2:
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    #r2 = r_value**2
    r2 = r2_score(x_clean, y_clean)

    x_seq = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_seq = slope * x_seq + intercept

    n = len(x_clean)
    t_crit = stats.t.ppf(0.975, df=n - 2)
    y_model = slope * x_clean + intercept
    s_err = np.sqrt(np.sum((y_clean - y_model) ** 2) / (n - 2))
    mean_x = np.mean(x_clean)
    Sxx = np.sum((x_clean - mean_x) ** 2)
    ci_bands = t_crit * s_err * np.sqrt(1.0 / n + (x_seq - mean_x) ** 2 / Sxx)

    ax.plot(x_seq, y_seq, color=line_color, linewidth=2, label="Fit")
    ax.fill_between(x_seq, y_seq - ci_bands, y_seq + ci_bands, color=fill_color, alpha=0.1)

    text_str = f"$R^2 = {r2:.3f}$\n$r = {r_value:.3f}$"
    ax.text(0.05, 0.92, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="white", edgecolor="0.8", alpha=0.8))

def plot_bland_altman(ax, m1, m2, title="Bland-Altman Agreement"):
    """Bland-Altman plot with improved visual cues."""
    mask = ~np.isnan(m1) & ~np.isnan(m2)
    d1, d2 = m1[mask], m2[mask]
    mean_val = (d1 + d2) / 2.0
    diff_val = d1 - d2

    md = np.mean(diff_val)
    sd = np.std(diff_val, ddof=1)

    ax.scatter(mean_val, diff_val, alpha=0.5, c="#8e44ad", s=35, edgecolors="white", linewidth=0.5)
    ax.axhline(md, color="#2c3e50", linestyle="-", linewidth=1.5)
    ax.axhline(md + 1.96 * sd, color="#c0392b", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(md - 1.96 * sd, color="#c0392b", linestyle="--", linewidth=1, alpha=0.7)

    # Annotations placed slightly outside data range
    x_pos = mean_val.max() + (mean_val.max() - mean_val.min()) * 0.05
    ax.text(x_pos, md, f"Mean\n{md:.2f}", va="center", color="#2c3e50", fontweight='bold')
    ax.text(x_pos, md + 1.96 * sd, f"+1.96 SD\n{md+1.96*sd:.2f}", va="center", color="#c0392b")
    ax.text(x_pos, md - 1.96 * sd, f"-1.96 SD\n{md-1.96*sd:.2f}", va="center", color="#c0392b")

    ax.set_xlabel("Average (BAc, BA_EC)")
    ax.set_ylabel("Difference (BAc - BA_EC)")
    ax.set_title(title, pad=15)

def plot_dashboard(df_sub, group_name, out_path):
    """Refined 3x2 Dashboard with optimized spacing and palette."""
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f"Biological Age Analysis: {group_name}", fontsize=20, fontweight='bold', y=0.98)

    # Palette setup
    c1, c2 = "#2980b9", "#e67e22" # Blue and Orange

    # 1. BAc vs CA
    ax1 = axs[0, 0]
    ax1.scatter(df_sub["Age"], df_sub["BAc"], alpha=0.6, c=c1, s=40, edgecolors="white")
    add_regression_stats(ax1, df_sub["Age"].values, df_sub["BAc"].values, line_color=c1, fill_color=c1)
    ax1.plot([-10, 150], [-10, 150], color="#7f8c8d", linestyle="--", alpha=0.6, label="Identity")
    force_square_axes(ax1, df_sub["Age"].values, df_sub["BAc"].values)
    ax1.set_title("PCA-Dubina (BAc) vs CA")
    ax1.set_xlabel("Chronological Age (CA)")
    ax1.set_ylabel("Predicted BAc")

    # 2. BA_EC vs CA
    ax2 = axs[0, 1]
    ax2.scatter(df_sub["Age"], df_sub["BA_EC"], alpha=0.6, c=c2, s=40, edgecolors="white")
    add_regression_stats(ax2, df_sub["Age"].values, df_sub["BA_EC"].values, line_color=c2, fill_color=c2)
    ax2.plot([-10, 150], [-10, 150], color="#7f8c8d", linestyle="--", alpha=0.6)
    force_square_axes(ax2, df_sub["Age"].values, df_sub["BA_EC"].values)
    ax2.set_title("KDM (BA_EC) vs CA")
    ax2.set_xlabel("Chronological Age (CA)")
    ax2.set_ylabel("Predicted BA_EC")

    # 3. Correlation between methods
    ax3 = axs[1, 0]
    ax3.scatter(df_sub["BAc"], df_sub["BA_EC"], alpha=0.6, c="#16a085", s=40, edgecolors="white")
    add_regression_stats(ax3, df_sub["BAc"].values, df_sub["BA_EC"].values, line_color="#16a085")
    ax3.plot([-10, 150], [-10, 150], color="#7f8c8d", linestyle="--", alpha=0.6)
    force_square_axes(ax3, df_sub["BAc"].values, df_sub["BA_EC"].values)
    ax3.set_title("Method Correlation: BAc vs BA_EC")
    ax3.set_xlabel("PCA-Dubina (BAc)")
    ax3.set_ylabel("KDM (BA_EC)")

    # 4. Bland-Altman
    plot_bland_altman(axs[1, 1], df_sub["BAc"].values, df_sub["BA_EC"].values)

    # 5. Histograms (Acceleration)
    ax5 = axs[2, 0]
    accel_bac = (df_sub["BAc"] - df_sub["Age"]).dropna()
    accel_kdm = (df_sub["BA_EC"] - df_sub["Age"]).dropna()
    
    sns.histplot(accel_bac, color=c1, alpha=0.3, label="PCA-BAc", kde=True, ax=ax5, element="step")
    sns.histplot(accel_kdm, color=c2, alpha=0.3, label="KDM-BA_EC", kde=True, ax=ax5, element="step")
    ax5.axvline(0, color="#2c3e50", linestyle="--", alpha=0.8)
    ax5.set_title("Distribution of Age Acceleration")
    ax5.set_xlabel("Acceleration (BA - CA)")
    ax5.legend()

    # 6. Accel vs Age (Bias check)
    ax6 = axs[2, 1]
    ax6.scatter(df_sub["Age"], accel_bac, alpha=0.5, c=c1, s=30, label="PCA-BAc")
    ax6.scatter(df_sub["Age"], accel_kdm, alpha=0.5, c=c2, s=30, label="KDM-BA_EC")
    ax6.axhline(0, color="#2c3e50", linestyle="--")
    ax6.set_title("Acceleration Bias Check")
    ax6.set_xlabel("Chronological Age (CA)")
    ax6.set_ylabel("Age Acceleration")
    ax6.legend()

    # Optimize spacing: h_pad and w_pad control the gap between subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4.0, w_pad=3.0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path

def plot_ba_results(pred_csv: Path, out_dir: Path, log=print):
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    required = {"Age", "BAc", "BA_EC"}
    if not required.issubset(df.columns):
        raise ValueError(f"Predictions file lacks needed columns: {required}")

    saved_files = []
    f_all = plot_dashboard(df, "All Groups", plots_dir / "BACalc_Combined_Palette.png")
    saved_files.append(f_all)
    log(f"Generated Dashboard: {f_all}")

    if "Group" in df.columns:
        for g in df["Group"].unique():
            sub = df[df["Group"] == g]
            if len(sub) >= 3:
                tag = sanitize_for_filename(str(g))
                f_g = plot_dashboard(sub, f"Group: {g}", plots_dir / f"BACalc_{tag}.png")
                saved_files.append(f_g)
    
    return {"plots_dir": plots_dir, "files": saved_files}
