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
    """
    Forces the X and Y axes to have the same limits and same number of ticks
    to create a perfect square plot with a 1:1 aspect ratio.
    """
    # 1. Determine common limits
    # Combine data to find global min/max
    all_data = np.concatenate([data_x, data_y])
    # Remove NaNs
    all_data = all_data[~np.isnan(all_data)]

    if len(all_data) == 0:
        # If no valid data, just set some defaults or skip
        ax.set_aspect("equal", adjustable="box")
        return

    d_min, d_max = all_data.min(), all_data.max()

    # Add 5% padding
    pad = (d_max - d_min) * 0.05
    if pad == 0:
        pad = 1.0  # Ensure some padding even if data is single value
        lim_min = d_min - pad
        lim_max = d_max + pad
    else:
        lim_min = d_min - pad
        lim_max = d_max + pad

    # 2. Apply limits to both axes
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # 3. Force aspect ratio to be equal (square)
    ax.set_aspect("equal", adjustable="box")

    # 4. Force same number of ticks
    locator = ticker.MaxNLocator(nbins=6)  # Approx 6 major ticks
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)


def add_regression_stats(ax, x, y, line_color="red", fill_color="red"):
    """
    Calculates regression, adds fit line with 95% Confidence Interval,
    and displays R2 and Pearson r on the plot.
    """
    # Drop NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return

    # Statistics
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r2 = r_value**2

    # Fit line data
    x_seq = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_seq = slope * x_seq + intercept

    # Confidence Interval Calculation
    n = len(x_clean)
    t_crit = stats.t.ppf(0.975, df=n - 2)  # 95% two-sided

    # Standard error of the estimate (residuals)
    y_model = slope * x_clean + intercept
    residuals = y_clean - y_model
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))

    mean_x = np.mean(x_clean)
    Sxx = np.sum((x_clean - mean_x) ** 2)

    # Calculate bands
    ci_bands = t_crit * s_err * np.sqrt(1.0 / n + (x_seq - mean_x) ** 2 / Sxx)

    # Plotting
    ax.plot(x_seq, y_seq, color=line_color, linewidth=2, linestyle="-", label="Fit")

    ax.fill_between(
        x_seq,
        y_seq - ci_bands,
        y_seq + ci_bands,
        color=fill_color,
        alpha=0.15,
        label="95% CI",
        edgecolor="none",
    )

    # Text Box stats
    text_str = (
        f"$R^2 = {r2:.3f}$\n$r = {r_value:.3f}$\n$P < {p_value:.3e}$"
        if p_value > 0
        else f"$R^2 = {r2:.3f}$\n$r = {r_value:.3f}$"
    )

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
    Bland-Altman plot with color.
    X = Average
    Y = Difference
    """
    mask = ~np.isnan(m1) & ~np.isnan(m2)
    d1, d2 = m1[mask], m2[mask]

    mean_val = (d1 + d2) / 2
    diff_val = d2 - d1  # (Predicted - Actual)

    md = np.mean(diff_val)
    sd = np.std(diff_val, ddof=1)

    ax.scatter(mean_val, diff_val, alpha=0.6, c="chocolate", s=30, edgecolors="none")

    ax.axhline(md, color="black", linestyle="-", linewidth=1.5)
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--", linewidth=1)
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--", linewidth=1)

    # Text for limits
    x_max = mean_val.max()
    offset_x = (
        (mean_val.max() - mean_val.min()) * 0.02
        if (mean_val.max() - mean_val.min()) > 0
        else 0.5
    )

    ax.text(
        x_max + offset_x,
        md + 1.96 * sd,
        f"+1.96 SD\n({md + 1.96*sd:.2f})",
        ha="left",
        va="center",
        fontsize=9,
    )

    ax.text(
        x_max + offset_x,
        md - 1.96 * sd,
        f"-1.96 SD\n({md - 1.96*sd:.2f})",
        ha="left",
        va="center",
        fontsize=9,
    )

    ax.text(
        x_max + offset_x, md, f"Mean\n({md:.2f})", ha="left", va="center", fontsize=9
    )

    ax.set_xlabel("Average (CA + BAc) / 2")
    ax.set_ylabel("Difference (BAc - CA)")
    ax.set_title(title)

    ax.set_box_aspect(1)  # Make the plot square


def plot_dashboard(df_sub, group_name, out_path):
    """
    Creates a 2x2 dashboard (Color, Square Plots).
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Analysis: {group_name}", fontsize=16)

    # ==========================================
    # 1. BAc vs Age (Corrected)
    # ==========================================
    ax1 = axs[0, 0]
    ax1.scatter(
        df_sub["Age"],
        df_sub["BAc"],
        alpha=0.6,
        c="blue",
        s=30,
        edgecolors="none",
        label="Data",
    )

    add_regression_stats(
        ax1,
        df_sub["Age"].values,
        df_sub["BAc"].values,
        line_color="darkblue",
        fill_color="blue",
    )

    ax1.plot([-100, 200], [-100, 200], "k--", alpha=0.5, label="Identity", linewidth=1)

    force_square_axes(ax1, df_sub["Age"].values, df_sub["BAc"].values)

    ax1.set_xlabel("Chronological Age (CA)")
    ax1.set_ylabel("Biological Age (BAc)")
    ax1.set_title("Corrected BA (BAc) vs CA")

    # ==========================================
    # 2. BA vs Age (Uncorrected)
    # ==========================================
    ax2 = axs[0, 1]
    ax2.scatter(
        df_sub["Age"],
        df_sub["BA"],
        alpha=0.6,
        c="green",
        s=30,
        edgecolors="none",
        label="Data",
    )

    add_regression_stats(
        ax2,
        df_sub["Age"].values,
        df_sub["BA"].values,
        line_color="darkgreen",
        fill_color="green",
    )
    ax2.plot([-100, 200], [-100, 200], "k--", alpha=0.5, linewidth=1)

    force_square_axes(ax2, df_sub["Age"].values, df_sub["BA"].values)

    ax2.set_xlabel("Chronological Age (CA)")
    ax2.set_ylabel("Biological Age (BA)")
    ax2.set_title("Uncorrected BA vs CA")

    # ==========================================
    # 3. Bland-Altman (BAc vs CA)
    # ==========================================
    plot_bland_altman(
        axs[1, 0],
        df_sub["Age"].values,
        df_sub["BAc"].values,
        title="Bland-Altman (BAc vs CA)",
    )

    # ==========================================
    # 4. Histogram of Acceleration
    # ==========================================
    ax4 = axs[1, 1]
    accel = (df_sub["BAc"] - df_sub["Age"]).dropna()

    ax4.hist(accel, bins=15, color="deepskyblue", edgecolor="lightblue", alpha=0.8)
    ax4.axvline(0, color="black", linestyle="--", linewidth=1.5)

    ax4.set_xlabel("Age Acceleration (BAc - CA)")
    ax4.set_ylabel("Frequency")
    ax4.set_title(
        f"Distribution of Acceleration\nMean={accel.mean():.2f}, SD={accel.std():.2f}"
    )
    ax4.set_box_aspect(1)  # Make the plot square

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_ba_results(pred_csv: Path, out_dir: Path, log=print):
    """main function for plotting ba results"""
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_csv)
    if not {"Age", "BA", "BAc"}.issubset(df.columns):
        raise ValueError("Predictions file lacks needed columns (Age, BA, BAc).")

    saved_files = []

    # Plot All Data Combined
    f_all = plot_dashboard(
        df, "All Groups Combined", plots_dir / "Dashboard_All_Groups.png"
    )
    saved_files.append(f_all)
    log(f"Saved plot: {f_all}")

    # Plot per Group if column exists and has more than 1 group
    if "Group" in df.columns:
        groups = df["Group"].unique()
        if len(groups) > 1:
            for g in groups:
                sub = df[df["Group"] == g]
                if len(sub) < 3:
                    log(f"Skipping plot for group {g} (not enough data)")
                    continue
                tag = sanitize_for_filename(str(g))
                f_g = plot_dashboard(
                    sub, f"Group: {g}", plots_dir / f"Dashboard_{tag}.png"
                )
                saved_files.append(f_g)
                log(f"Saved plot: {f_g}")

    return {"plots_dir": plots_dir, "files": saved_files}
