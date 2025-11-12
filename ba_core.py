from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats

EPS = 1e-12


def to_numeric(df: pd.DataFrame, cols) -> None:
    """change values to numeric"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def zscore(X: np.ndarray):
    """computes zscore here"""
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z_values = (X - mu) / sd
    return Z_values, mu, sd


def t_scale(bas: np.ndarray, ca: np.ndarray, ddof: int = 0):
    """performs t-scale calculations"""
    bas_mu = np.nanmean(bas)
    bas_sd = np.nanstd(bas, ddof=ddof) or 1.0
    ca_mu = np.nanmean(ca)
    ca_sd = np.nanstd(ca, ddof=ddof)
    return (bas - bas_mu) * (ca_sd / bas_sd) + ca_mu


def dubina_correct(BA: np.ndarray, CA: np.ndarray):
    """performs dubina correction here"""
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


def write_equations_txt(
    out_path, group, biom_cols, wn, w0, bas_mu, bas_sd, ca_mean, ca_sd, b
):
    """
    BA  = alpha * BAS + beta
    BAc = BA + (CA - mu_ca) * gamma

    BAS_g = w0 + sum(w_i * x_i)
    BA_g  = (beta + alpha*w0) + sum((alpha*w_i) * x_i)
    BAc_g = (beta + alpha*w0 - gamma*mu_ca) + gamma*CA + sum((alpha*w_i) * x_i)
    """
    alpha = ca_sd / (bas_sd if bas_sd != 0 else 1.0)
    beta = ca_mean - alpha * bas_mu
    gamma = 1.0 - b
    mu_ca = ca_mean

    ba_intercept = beta + alpha * w0
    bac_intercept = ba_intercept - gamma * mu_ca

    lines = []
    lines.append("=" * 88)
    lines.append(f"Group: {group}")
    lines.append(
        f"T-scale: alpha={alpha:.10g}, beta={beta:.10g}   (BA = alpha*BAS + beta)"
    )
    lines.append(
        f"Dubina : gamma={gamma:.10g}, mu_CA={mu_ca:.10g} (BAc = BA + (CA - mu_CA)*gamma)"
    )
    lines.append(
        f"Dataset stats used: bas_mu={bas_mu:.10g}, bas_sd={bas_sd:.10g}, "
        f"ca_mean={ca_mean:.10g}, ca_sd={ca_sd:.10g}, b={b:.10g}"
    )
    lines.append("")
    bas_terms = " + ".join(
        [f"({wn[i]:.10g}*{biom_cols[i]})" for i in range(len(biom_cols))]
    )
    lines.append("BAS (raw linear form):")
    lines.append(f"  BAS_{group} = {w0:.10g} + {bas_terms}")
    lines.append("")
    ba_terms = " + ".join(
        [f"({(alpha*wn[i]):.10g}*{biom_cols[i]})" for i in range(len(biom_cols))]
    )
    lines.append("BA (expanded from T-scale):")
    lines.append(f"  BA_{group} = {ba_intercept:.10g} + {ba_terms}")
    lines.append("")
    bac_terms = ba_terms
    lines.append("BAc (expanded from Dubina correction):")
    lines.append(
        f"  BAc_{group} = {bac_intercept:.10g} + ({gamma:.10g}*CA) + {bac_terms}"
    )
    lines.append("")

    out_path = Path(out_path)
    # write to txt file
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def run_ba_pipeline(
    df: pd.DataFrame,
    age_col: str,
    biom_cols: list[str],
    split_col: str | None,
    out_dir: Path,
    log=print,
):
    """uses all function to run the BA calculations aka pipeline"""
    out_dir.mkdir(parents=True, exist_ok=True)

    to_numeric(df, [age_col] + biom_cols + ([split_col] if split_col else []))

    # binary split or single group
    if split_col:
        uniq = [v for v in pd.Series(df[split_col].dropna().unique()).tolist()]
        if len(uniq) != 2:
            raise ValueError(
                f"Selected split column '{split_col}' must be binary (found {len(uniq)} unique)."
            )
        groups = [
            (f"{split_col}={repr(uniq[0])}", df[split_col] == uniq[0]),
            (f"{split_col}={repr(uniq[1])}", df[split_col] == uniq[1]),
        ]
    else:
        groups = [("All", pd.Series(True, index=df.index))]

    pred_rows = []
    load_rows = []
    coef_rows = []

    eq_path = out_dir / "ba_equations.txt"
    if eq_path.exists():
        eq_path.unlink()

    for gname, mask in groups:
        sub = df.loc[mask].copy()
        if sub.empty:
            log(f"Group '{gname}' is empty, skipping.")
            continue

        # median impute within group for selected biomaarkers
        for c in biom_cols:
            med = float(np.nanmedian(sub[c]))
            sub[c] = sub[c].fillna(med)

        X = sub[biom_cols].to_numpy(dtype=float)
        Z_values, mu_x, sd_x = zscore(X)

        # pca and select PC1
        pca = PCA(n_components=1, svd_solver="full")
        pca.fit(Z_values)
        pc1 = pca.components_[0].copy()

        ca = sub[age_col].to_numpy(dtype=float)

        # align sign so PC1 increases with age
        r = np.corrcoef(Z_values @ pc1, ca)[0, 1]
        if np.isnan(r):
            r = 1.0
        if r < 0:
            pc1 *= -1

        wn = pc1 / sd_x
        w0 = float(-np.sum(wn * mu_x))
        BAS = X.dot(wn) + w0

        # T-scale
        ca_mean = float(np.nanmean(ca))
        ca_sd = float(np.nanstd(ca, ddof=0))
        bas_mu = float(np.nanmean(BAS))
        bas_sd = float(np.nanstd(BAS, ddof=0)) or 1.0

        BA = t_scale(BAS, ca)
        BAc, b, _, _ = dubina_correct(BA, ca)

        # equations
        write_equations_txt(
            out_path=eq_path,
            group=gname,
            biom_cols=biom_cols,
            wn=wn,
            w0=w0,
            bas_mu=bas_mu,
            bas_sd=bas_sd,
            ca_mean=ca_mean,
            ca_sd=ca_sd,
            b=b,
        )

        # outputs
        pred_rows.append(
            pd.DataFrame(
                {
                    "Group": gname,
                    "Age": ca,
                    "BA": BA,
                    "BAc": BAc,
                    "Accel": BA - ca,
                    "Accel_corr": BAc - ca,
                },
                index=sub.index,
            )
        )

        load_rows.extend(
            {
                "Group": gname,
                "variable": v,
                "loading_PC1": float(ld),
                "mean_used": float(mu),
                "sd_used": float(sd),
            }
            for v, ld, mu, sd in zip(biom_cols, pc1, mu_x, sd_x)
        )

        coef_rows.extend(
            {"Group": gname, "variable": v, "coef_w": float(w)}
            for v, w in zip(biom_cols, wn)
        )
        coef_rows.append(
            {"Group": gname, "variable": "intercept_w0", "coef_w": float(w0)}
        )

        log(f"Finished group '{gname}': n={len(sub)}, b={b:.5g}, bas_sd={bas_sd:.5g}")

    if not pred_rows:
        raise ValueError(
            "No rows available to compute BA (check split column or data)."
        )

    res = pd.concat(pred_rows)
    pred_path = out_dir / "ba_predictions.csv"
    load_path = out_dir / "pca_loadings.csv"
    coef_path = out_dir / "ba_coefficients.csv"

    pd.DataFrame(load_rows).to_csv(load_path, index=False)
    pd.DataFrame(coef_rows).to_csv(coef_path, index=False)
    res.to_csv(pred_path, index=False)

    return {
        "predictions": pred_path,
        "loadings": load_path,
        "coefs": coef_path,
        "equations": eq_path,
    }
