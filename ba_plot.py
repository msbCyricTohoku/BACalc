from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats


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
    pad = (d_max - d_min) * 0.05
    pad = 1.0 if pad == 0 else pad
    lim_min, lim_max = d_min - pad, d_max + pad

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")

    locator = ticker.MaxNLocator(nbins=6)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)


def add_regression_stats(ax, x, y, line_color="red", fill_color="red"):
    """Adds regression line, CI, and R2/r stats."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 2:
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r2 = r_value**2

    x_seq = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_seq = slope * x_seq + intercept

    n = len(x_clean)
    t_crit = stats.t.ppf(0.975, df=n - 2)

    y_model = slope * x_clean + intercept
    s_err = np.sqrt(np.sum((y_clean - y_model) ** 2) / (n - 2))
    mean_x = np.mean(x_clean)
    Sxx = np.sum((x_clean - mean_x) ** 2)

    ci_bands = t_crit * s_err * np.sqrt(1.0 / n + (x_seq - mean_x) ** 2 / Sxx)

    ax.plot(x_seq, y_seq, color=line_color, linewidth=2, linestyle="-", label="Fit")
    ax.fill_between(
        x_seq,
        y_seq - ci_bands,
        y_seq + ci_bands,
        color=fill_color,
        alpha=0.15,
        edgecolor="none",
    )

    text_str = f"$R^2 = {r2:.3f}$\n$r = {r_value:.3f}$"
    props = dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)
    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )


def plot_bland_altman(ax, m1, m2, title="Bland-Altman"):
    """
    Bland-Altman plot (Purple style).
    X = Average, Y = Difference (m1 - m2)
    """
    mask = ~np.isnan(m1) & ~np.isnan(m2)
    d1, d2 = m1[mask], m2[mask]

    mean_val = (d1 + d2) / 2.0
    diff_val = d1 - d2  # Method 1 - Method 2

    md = np.mean(diff_val)
    sd = np.std(diff_val, ddof=1)

    ax.scatter(mean_val, diff_val, alpha=0.6, c="purple", s=40, edgecolors="none")

    ax.axhline(md, color="black", linestyle="-", linewidth=1.5)
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--", linewidth=1)
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--", linewidth=1)

    # Text for limits
    x_max = mean_val.max()
    offset_x = (x_max - mean_val.min()) * 0.02

    ax.text(
        x_max + offset_x,
        md + 1.96 * sd,
        f"+1.96 SD\n({md+1.96*sd:.2f})",
        ha="left",
        va="center",
        fontsize=9,
    )
    ax.text(
        x_max + offset_x,
        md - 1.96 * sd,
        f"-1.96 SD\n({md-1.96*sd:.2f})",
        ha="left",
        va="center",
        fontsize=9,
    )
    ax.text(
        x_max + offset_x, md, f"Mean\n({md:.2f})", ha="left", va="center", fontsize=9
    )

    ax.set_xlabel("Average (BAc + BA_EC) / 2")
    ax.set_ylabel("Difference (BAc - BA_EC)")
    ax.set_title(title)
    ax.set_box_aspect(1)


def plot_dashboard(df_sub, group_name, out_path):
    """
    Creates a 3x2 dashboard:
    Row 1: BAc vs Age | BA_EC vs Age
    Row 2: BAc vs BA_EC (Correlation) | Bland-Altman (BAc vs BA_EC)
    Row 3: Histograms (Overlay) | Accel vs Age (Overlay)
    """
    fig, axs = plt.subplots(3, 2, figsize=(14, 20))
    fig.suptitle(f"Analysis: {group_name}", fontsize=18)

    # ==========================================
    # Row 1: Basic Models vs CA
    # ==========================================
    # Ax1: PCA-Dubina (BAc)
    ax1 = axs[0, 0]
    ax1.scatter(
        df_sub["Age"],
        df_sub["BAc"],
        alpha=0.6,
        c="blue",
        s=40,
        edgecolors="none",
        label="PCA-Dubina",
    )
    add_regression_stats(
        ax1,
        df_sub["Age"].values,
        df_sub["BAc"].values,
        line_color="darkblue",
        fill_color="blue",
    )
    ax1.plot([-100, 200], [-100, 200], "k--", alpha=0.5, label="Identity")
    force_square_axes(ax1, df_sub["Age"].values, df_sub["BAc"].values)
    ax1.set_xlabel("Chronological Age (CA)")
    ax1.set_ylabel("PCA-Dubina BAc")
    ax1.set_title("Method 1: PCA-Dubina vs CA")
    ax1.legend(loc="lower right")

    # Ax2: KDM (BA_EC)
    ax2 = axs[0, 1]
    ax2.scatter(
        df_sub["Age"],
        df_sub["BA_EC"],
        alpha=0.6,
        c="crimson",
        s=40,
        edgecolors="none",
        label="KDM",
    )
    add_regression_stats(
        ax2,
        df_sub["Age"].values,
        df_sub["BA_EC"].values,
        line_color="darkred",
        fill_color="crimson",
    )
    ax2.plot([-100, 200], [-100, 200], "k--", alpha=0.5)
    force_square_axes(ax2, df_sub["Age"].values, df_sub["BA_EC"].values)
    ax2.set_xlabel("Chronological Age (CA)")
    ax2.set_ylabel("KDM BA_EC")
    ax2.set_title("Method 2: KDM vs CA")
    ax2.legend(loc="lower right")

    # ==========================================
    # Row 2: Comparison
    # ==========================================
    # Ax3: Scatter Correlation
    ax3 = axs[1, 0]
    ax3.scatter(
        df_sub["BAc"],
        df_sub["BA_EC"],
        alpha=0.6,
        c="chocolate",
        s=40,
        edgecolors="none",
    )
    add_regression_stats(
        ax3,
        df_sub["BAc"].values,
        df_sub["BA_EC"].values,
        line_color="coral",
        fill_color="chocolate",
    )
    ax3.plot([-100, 200], [-100, 200], "k--", alpha=0.5, label="1:1 Line")
    force_square_axes(ax3, df_sub["BAc"].values, df_sub["BA_EC"].values)
    ax3.set_xlabel("PCA-Dubina (BAc)")
    ax3.set_ylabel("KDM (BA_EC)")
    ax3.set_title("Correlation: BAc vs BA_EC")

    # Ax4: Bland-Altman
    ax4 = axs[1, 1]
    plot_bland_altman(
        ax4,
        df_sub["BAc"].values,
        df_sub["BA_EC"].values,
        title="Bland-Altman: BAc vs BA_EC",
    )

    # ==========================================
    # Row 3: Analysis
    # ==========================================
    # Ax5: Histograms
    ax5 = axs[2, 0]
    accel_bac = (df_sub["BAc"] - df_sub["Age"]).dropna()
    accel_kdm = (df_sub["BA_EC"] - df_sub["Age"]).dropna()

    ax5.hist(
        accel_bac,
        bins=20,
        color="deepskyblue",
        alpha=0.35,
        label="PCA-BAc",
        density=True,
        edgecolor="deepskyblue",
    )
    ax5.hist(
        accel_kdm,
        bins=20,
        color="chocolate",
        alpha=0.35,
        label="KDM-BA_EC",
        density=True,
        edgecolor="chocolate",
    )

    # KDE Lines
    try:
        if len(accel_bac) > 5 and len(accel_kdm) > 5:
            kde_bac = stats.gaussian_kde(accel_bac)
            kde_kdm = stats.gaussian_kde(accel_kdm)
            x_range = np.linspace(
                min(accel_bac.min(), accel_kdm.min()),
                max(accel_bac.max(), accel_kdm.max()),
                100,
            )
            ax5.plot(x_range, kde_bac(x_range), color="deepskyblue", lw=2)
            ax5.plot(x_range, kde_kdm(x_range), color="chocolate", lw=2)
    except:
        pass
    ax5.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Age Acceleration (BA - CA)")
    ax5.set_ylabel("Density")
    ax5.set_title("Accel Distribution Overlay")
    ax5.legend()
    ax5.set_box_aspect(1)

    # Ax6: Acceleration vs Age (Check for bias)
    ax6 = axs[2, 1]
    ax6.scatter(
        df_sub["Age"],
        accel_bac,
        alpha=0.5,
        c="blue",
        s=30,
        edgecolors="none",
        label="PCA-BAc",
    )
    ax6.scatter(
        df_sub["Age"],
        accel_kdm,
        alpha=0.5,
        c="crimson",
        s=30,
        edgecolors="none",
        label="KDM-BA_EC",
    )
    ax6.axhline(0, color="black", linestyle="--")
    ax6.set_xlabel("Chronological Age (CA)")
    ax6.set_ylabel("Age Acceleration")
    ax6.set_title("Age Acceleration vs Age")
    ax6.legend()
    ax6.set_box_aspect(1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300)
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

    # All Data
    f_all = plot_dashboard(df, "All Groups", plots_dir / "BACalc_All_Combined.png")
    saved_files.append(f_all)
    log(f"Saved plot: {f_all}")

    # Per Group
    if "Group" in df.columns:
        groups = df["Group"].unique()
        if len(groups) > 1:
            for g in groups:
                sub = df[df["Group"] == g]
                if len(sub) < 3:
                    continue
                tag = sanitize_for_filename(str(g))
                f_g = plot_dashboard(
                    sub, f"Group: {g}", plots_dir / f"BACalc_{tag}.png"
                )
                saved_files.append(f_g)
                log(f"Saved plot: {f_g}")

    return {"plots_dir": plots_dir, "files": saved_files}
