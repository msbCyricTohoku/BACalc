from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats

EPS = 1e-12
MIN_S_VALUE = 1e-6
MIN_R_CHAR = 1e-6
S2_BA_FLOOR = 0.1


def to_numeric(df: pd.DataFrame, cols) -> None:
    """change values to numeric"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def zscore(x_array: np.ndarray):
    """computes zscore here"""
    mu = np.nanmean(x_array, axis=0)
    sd = np.nanstd(x_array, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    z_values = (x_array - mu) / sd
    return z_values, mu, sd


# ==========================================
# PCA-Dubina
# ==========================================
def t_scale(bas: np.ndarray, ca: np.ndarray, ddof: int = 0):
    bas_mu = np.nanmean(bas)
    bas_sd = np.nanstd(bas, ddof=ddof) or 1.0
    ca_mu = np.nanmean(ca)
    ca_sd = np.nanstd(ca, ddof=ddof)
    return (bas - bas_mu) * (ca_sd / bas_sd) + ca_mu


def dubina_correct(BA: np.ndarray, CA: np.ndarray):
    ca_mean = np.nanmean(CA)
    var_ca = np.nanvar(CA, ddof=0) + EPS
    cov_ba_ca = np.nanmean((BA - np.nanmean(BA)) * (CA - ca_mean))
    b = cov_ba_ca / var_ca
    return (
        BA + (CA - ca_mean) * (1.0 - b),
        float(b),
        float(ca_mean),
        float(np.nanstd(CA, ddof=0)),
    )


# ==========================================
# KDM
# ==========================================
def train_kdm_params(df_sub, age_col, biom_cols):
    """
    Calculates k (slope), q (intercept), s (rmse), r (corr) for each biomarker.
    Also calculates r_char and S2_BA.
    """
    X_age = df_sub[[age_col]].values

    k_vals, q_vals, s_vals, r_vals = [], [], [], []

    for biom in biom_cols:
        y = df_sub[[biom]].values
        # Avoid constant values
        if np.var(y) < 1e-10:
            k, q, s, r = 0.0, 0.0, 1.0, 0.0
        else:
            mdl = LinearRegression().fit(X_age, y)
            k = mdl.coef_[0][0]
            q = mdl.intercept_[0]
            y_pred = mdl.predict(X_age)
            s = np.sqrt(mean_squared_error(y, y_pred))
            r_stat, _ = stats.pearsonr(df_sub[age_col], df_sub[biom])
            r = r_stat if not np.isnan(r_stat) else 0.0

        k_vals.append(k)
        q_vals.append(q)
        s_vals.append(s if s > MIN_S_VALUE else 1.0)
        r_vals.append(r)

    # Calculate r_char
    num, den = 0, 0
    for r in r_vals:
        if abs(r) >= 1 or abs(r) < MIN_R_CHAR:
            continue
        sqrt_term = np.sqrt(1 - r**2)
        num += (r**2) / sqrt_term
        den += r / sqrt_term

    r_char = (num / den) if abs(den) > 1e-9 else 0.0
    r_char = np.clip(r_char, -0.9999, 0.9999)

    # Calculate S2_BA (population variance approach)
    n = len(df_sub)
    m = len(biom_cols)
    ba_e_vals = []

    for i in range(n):
        num_kdm, den_kdm = 0, 0
        for j in range(m):
            val = df_sub[biom_cols[j]].iloc[i]
            if abs(k_vals[j]) < 1e-9:
                continue

            weight = (k_vals[j] / s_vals[j]) ** 2
            age_equiv = (val - q_vals[j]) / k_vals[j]

            num_kdm += age_equiv * weight
            den_kdm += weight

        ba_e_vals.append(num_kdm / den_kdm if den_kdm > 1e-9 else np.nan)

    ba_e_arr = np.array(ba_e_vals)
    ca_arr = df_sub[age_col].values

    diff = ba_e_arr - ca_arr
    diff = diff[~np.isnan(diff)]
    term1 = np.var(diff, ddof=0) if len(diff) > 1 else 10.0

    ca_range = np.max(ca_arr) - np.min(ca_arr)
    var_ca_approx = ((ca_range**2) / 12.0) if ca_range > 1e-9 else 0.0

    try:
        if abs(r_char) > MIN_R_CHAR:
            factor_A = (1.0 - r_char**2) / (r_char**2)
            factor_B = var_ca_approx / m
            term2 = factor_A * factor_B
        else:
            term2 = 0
    except:
        term2 = 0

    s2_ba = max(S2_BA_FLOOR, term1 - term2)

    return {
        "k": k_vals,
        "q": q_vals,
        "s": s_vals,
        "r": r_vals,
        "r_char": r_char,
        "s2_ba": s2_ba,
    }


def calculate_kdm_scores(row, biom_cols, params, ca_val):
    """Calculates BA_E and BA_EC for a single row"""
    num, den = 0, 0
    k, q, s = params["k"], params["q"], params["s"]
    s2_ba = params["s2_ba"]

    # BA_E Calculation
    for j, biom in enumerate(biom_cols):
        val = row[biom]
        if np.isnan(val) or abs(k[j]) < 1e-9:
            continue

        weight = (k[j] / s[j]) ** 2
        char_age = (val - q[j]) / k[j]

        num += char_age * weight
        den += weight

    if den < 1e-9:
        return np.nan, np.nan

    ba_e = num / den

    # BA_EC Calculation (Bayesian update)
    num_ec = num + (ca_val / s2_ba)
    den_ec = den + (1.0 / s2_ba)

    ba_ec = num_ec / den_ec

    return ba_e, ba_ec


def write_combined_equations(
    out_path, group, biom_cols, wn, w0, bas_mu, bas_sd, ca_mean, ca_sd, b, kdm_params
):
    """Writes explicit linear equations for PCA and Parameters for KDM."""

    # --- PCA Calcs ---
    alpha = ca_sd / (bas_sd if bas_sd != 0 else 1.0)
    beta = ca_mean - alpha * bas_mu
    gamma = 1.0 - b
    mu_ca = ca_mean

    ba_intercept = beta + alpha * w0
    bac_intercept = ba_intercept - gamma * mu_ca

    lines = []
    lines.append("=" * 88)
    lines.append(f"GROUP: {group}")
    lines.append("=" * 88)
    lines.append("")
    lines.append("METHOD 1: PCA-DUBINA (BAc)")
    lines.append("-" * 40)
    lines.append("  BAS = Raw Score | BA = T-Scale | BAc = Corrected")
    lines.append("")

    # BAS Equation
    lines.append("1. BAS Equation:")
    terms_bas = [f"({wn[i]:.5g} * {biom_cols[i]})" for i in range(len(biom_cols))]
    lines.append(f"  BAS = {w0:.5g} + " + " + ".join(terms_bas))
    lines.append("")

    # BA Equation
    lines.append("2. BA (T-Scale) Equation:")
    terms_ba = [
        f"({(alpha*wn[i]):.5g} * {biom_cols[i]})" for i in range(len(biom_cols))
    ]
    lines.append(f"  BA = {ba_intercept:.5g} + " + " + ".join(terms_ba))
    lines.append(f"  ( Derived: BA = {alpha:.5g} * BAS + {beta:.5g} )")
    lines.append("")

    # BAc Equation
    lines.append("3. BAc (Dubina) Equation:")
    lines.append(
        f"  BAc = {bac_intercept:.5g} + ({gamma:.5g} * CA) + " + " + ".join(terms_ba)
    )
    lines.append(f"  ( Derived: BAc = BA + (CA - {mu_ca:.5g}) * {gamma:.5g} )")
    lines.append("")

    lines.append("METHOD 2: KDM (BA_EC)")
    lines.append("-" * 40)
    lines.append(f"  Correction Factor S^2_BA = {kdm_params['s2_ba']:.5g}")
    lines.append("")
    lines.append(
        f"{'Biomarker':<20} | {'k (slope)':<10} | {'q (int)':<10} | {'s (rmse)':<10} | {'Weight':<12}"
    )
    lines.append("-" * 75)

    sum_weights = 0
    for i, biom in enumerate(biom_cols):
        k = kdm_params["k"][i]
        q = kdm_params["q"][i]
        s = kdm_params["s"][i]
        w = (k / s) ** 2 if s > 0 else 0
        sum_weights += w
        lines.append(f"{biom:<20} | {k:<10.4g} | {q:<10.4g} | {s:<10.4g} | {w:<12.4g}")

    lines.append("-" * 75)
    lines.append(f"Total Weight (Sum W) = {sum_weights:.5g}")
    lines.append("")
    lines.append("4. BA_E Formula:")
    lines.append("  BA_E = Sum( Weight_i * (Biomarker_i - q_i) / k_i ) / Sum(Weight_i)")
    lines.append("")
    lines.append("5. BA_EC Formula:")
    lines.append(
        f"  BA_EC = ( (BA_E * {sum_weights:.5g}) + (CA / {kdm_params['s2_ba']:.5g}) ) / ( {sum_weights:.5g} + (1 / {kdm_params['s2_ba']:.5g}) )"
    )
    lines.append("\n\n")

    out_path = Path(out_path)
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def calculate_stats(df_res: pd.DataFrame, group_name: str):
    """Calculates R2, RMSE, MAE, Pearson r."""
    stats_list = []
    targets = ["BA", "BAc", "BA_E", "BA_EC"]

    for metric in targets:
        if metric not in df_res.columns:
            continue

        y_true = df_res["Age"]
        y_pred = df_res[metric]
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_t, y_p = y_true[mask], y_pred[mask]

        if len(y_t) < 2:
            continue

        r2 = r2_score(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        pearson_r, _ = stats.pearsonr(y_t, y_p)

        stats_list.append(
            {
                "Group": group_name,
                "Metric_Type": metric,
                "R2": r2,
                "Pearson_r": pearson_r,
                "RMSE": rmse,
                "MAE": mae,
                "N": len(y_t),
            }
        )
    return stats_list


def run_ba_pipeline(
    df: pd.DataFrame,
    age_col: str,
    biom_cols: list[str],
    split_col: str | None,
    out_dir: Path,
    log=print,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    to_numeric(df, [age_col] + biom_cols + ([split_col] if split_col else []))

    if split_col:
        uniq = [v for v in pd.Series(df[split_col].dropna().unique()).tolist()]
        if len(uniq) != 2:
            raise ValueError(f"Split col '{split_col}' must be binary.")
        groups = [
            (f"{split_col}={repr(uniq[0])}", df[split_col] == uniq[0]),
            (f"{split_col}={repr(uniq[1])}", df[split_col] == uniq[1]),
        ]
    else:
        groups = [("All", pd.Series(True, index=df.index))]

    pred_rows = []
    stats_rows = []
    load_rows = []

    eq_path = out_dir / "ba_equations.txt"
    if eq_path.exists():
        eq_path.unlink()

    for gname, mask in groups:
        sub = df.loc[mask].copy()
        if sub.empty or len(sub) < 5:
            log(f"Group '{gname}' insufficient data, skipping.")
            continue

        # Median impute
        for c in biom_cols:
            sub[c] = sub[c].fillna(sub[c].median())

        # --- METHOD 1: PCA-Dubina ---
        x_array = sub[biom_cols].to_numpy(dtype=float)
        z_values, mu_x, sd_x = zscore(x_array)

        pca = PCA(n_components=1, svd_solver="full")
        pca.fit(z_values)
        pc1 = pca.components_[0].copy()

        ca = sub[age_col].to_numpy(dtype=float)
        r = np.corrcoef(z_values @ pc1, ca)[0, 1]
        if r < 0:
            pc1 *= -1

        wn = pc1 / sd_x
        w0 = float(-np.sum(wn * mu_x))
        BAS = x_array.dot(wn) + w0

        BA = t_scale(BAS, ca)
        BAc, b, ca_mean, ca_sd = dubina_correct(BA, ca)

        load_rows.extend(
            {
                "Group": gname,
                "variable": v,
                "PC1_Weight": float(w),
                "PC1_Loading": float(l),
            }
            for v, w, l in zip(biom_cols, wn, pc1)
        )

        # --- METHOD 2: KDM ---
        kdm_params = train_kdm_params(sub, age_col, biom_cols)
        ba_e_list, ba_ec_list = [], []
        for idx, row in sub.iterrows():
            val_e, val_ec = calculate_kdm_scores(
                row, biom_cols, kdm_params, row[age_col]
            )
            ba_e_list.append(val_e)
            ba_ec_list.append(val_ec)

        # --- OUTPUTS ---
        bas_mu = float(np.nanmean(BAS))
        bas_sd = float(np.nanstd(BAS, ddof=0))

        write_combined_equations(
            eq_path,
            gname,
            biom_cols,
            wn,
            w0,
            bas_mu,
            bas_sd,
            ca_mean,
            ca_sd,
            b,
            kdm_params,
        )

        sub_res = pd.DataFrame(
            {
                "Group": gname,
                "Age": ca,
                "BAS": BAS,
                "BA": BA,
                "BAc": BAc,
                "BA_E": ba_e_list,
                "BA_EC": ba_ec_list,
                "Accel_BAc": BAc - ca,
                "Accel_BA_EC": np.array(ba_ec_list) - ca,
            },
            index=sub.index,
        )

        pred_rows.append(sub_res)
        stats_rows.extend(calculate_stats(sub_res, gname))

        log(f"Finished group '{gname}': n={len(sub)}")

    if not pred_rows:
        raise ValueError("No rows available.")

    res = pd.concat(pred_rows)
    pred_path = out_dir / "ba_predictions.csv"
    stats_path = out_dir / "ba_stats.csv"
    load_path = out_dir / "pca_loadings.csv"

    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    pd.DataFrame(load_rows).to_csv(load_path, index=False)
    res.to_csv(pred_path, index=False)

    return {
        "predictions": pred_path,
        "stats": stats_path,
        "equations": eq_path,
        "loadings": load_path,
    }
