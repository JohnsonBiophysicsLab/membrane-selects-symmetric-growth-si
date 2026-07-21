from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
RESULTS = ROOT / "results"
INPUT = RESULTS / "cylindrical_symmetry_candidate_trajectories.csv"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "mathtext.fontset": "cm",
        "axes.labelsize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.linewidth": 1.1,
    }
)


def mark_iqr_outliers(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    q1 = group["CylindricalAsymmetry"].quantile(0.25)
    q3 = group["CylindricalAsymmetry"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    group["OutlierLowerBound"] = lower
    group["OutlierUpperBound"] = upper
    group["MetricOutlierIQR"] = (group["CylindricalAsymmetry"] < lower) | (
        group["CylindricalAsymmetry"] > upper
    )
    return group


def bootstrap_joint_minimum(data: pd.DataFrame, n_bootstrap: int = 10000) -> pd.DataFrame:
    rows = []
    for (ecc, rigidity), subset in data.groupby(["Ecc", "Rigidity"], dropna=False):
        subset = subset[np.isfinite(subset["CylindricalAsymmetry"])].reset_index(drop=True)
        if subset.empty:
            continue
        observed = subset.loc[subset["Min_Etot"].idxmin()].copy()
        values = subset[["Min_Etot", "CylindricalAsymmetry"]].to_numpy(float)
        rng = np.random.default_rng(20260616 + int(round(1000 * float(ecc))) + int(round(float(rigidity))))
        sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        resampled = values[sample_indices]
        min_indices = np.argmin(resampled[:, :, 0], axis=1)
        selected_metrics = resampled[np.arange(n_bootstrap), min_indices, 1]
        rows.append(
            {
                "Ecc": float(ecc),
                "Rigidity": float(rigidity),
                "RunsAfterOutlierExclusion": int(len(subset)),
                "RunID": int(observed["RunID"]),
                "RelaxArea": float(observed["RelaxArea"]),
                "Min_Etot": float(observed["Min_Etot"]),
                "CylindricalAsymmetry": float(observed["CylindricalAsymmetry"]),
                "StdErrCylindricalAsymmetry": float(np.std(selected_metrics, ddof=1)),
                "BootstrapMetricP025": float(np.percentile(selected_metrics, 2.5)),
                "BootstrapMetricP975": float(np.percentile(selected_metrics, 97.5)),
            }
        )
    return pd.DataFrame(rows).sort_values(["Ecc", "Rigidity"]).reset_index(drop=True)


def plot_summary(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 3.9))
    styles = {
        0.0: {"fmt": "o", "color": "black", "label": "0"},
        0.866: {"fmt": "s", "color": "0.45", "label": "0.866"},
    }
    for ecc, style in styles.items():
        data = summary[np.isclose(summary["Ecc"], ecc)].sort_values("Rigidity")
        if data.empty:
            continue
        ax.errorbar(
            data["Rigidity"],
            data["CylindricalAsymmetry"],
            yerr=data["StdErrCylindricalAsymmetry"],
            fmt=style["fmt"],
            color=style["color"],
            ecolor=style["color"],
            capsize=5.5,
            capthick=1.8,
            elinewidth=1.4,
            markersize=6,
            label=style["label"],
        )
    ax.set_xscale("log")
    ax.set_xlim(8, 1200)
    ax.set_xlabel(r"Protein Lattice Rigidity (pN$\cdot$nm or pN/nm)")
    ax.set_ylabel("Cylindrical Asymmetry")
    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=1.0)
    legend = ax.legend(title="Eccentricity", frameon=True, fancybox=False, loc="best")
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        out = PLOTS / f"cylindrical_symmetry_vs_rigidity_iqr_outliers_removed.{suffix}"
        fig.savefig(out, dpi=450 if suffix == "png" else None, bbox_inches="tight")
        print(out)


def main() -> None:
    data = pd.read_csv(INPUT)
    marked = data.groupby(["Ecc", "Rigidity"], group_keys=False).apply(mark_iqr_outliers)
    marked.to_csv(RESULTS / "cylindrical_symmetry_candidate_trajectories_iqr_outliers.csv", index=False)
    filtered = marked[~marked["MetricOutlierIQR"]].copy()
    summary = bootstrap_joint_minimum(filtered)
    summary.to_csv(RESULTS / "cylindrical_symmetry_surface_summary_iqr_outliers_removed.csv", index=False)
    plot_summary(summary)


if __name__ == "__main__":
    main()
