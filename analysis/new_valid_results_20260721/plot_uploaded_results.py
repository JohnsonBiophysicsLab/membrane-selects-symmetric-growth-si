#!/usr/bin/env python3
"""Replot compact July 2026 valid-result figures from repository data."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "new_valid_results_20260721"
OUT = ROOT / "analysis" / "new_valid_results_20260721" / "recomputed_outputs"
OUT.mkdir(parents=True, exist_ok=True)

RIGIDITIES = [10.0, 31.6, 100.0, 316.0, 1000.0]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "svg.fonttype": "none",
        }
    )


def save(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_rigidity_energy_curvature() -> None:
    folder = DATA / "rigidity_energy_curvature"
    e0 = pd.read_csv(folder / "a30b30_min_etot_old_cap_grouped_minima.csv").sort_values("Rigidity")
    e0866 = pd.read_csv(folder / "a30b15_min_etot_old_cap_grouped_minima.csv").sort_values("Rigidity")

    fig, (ax_e, ax_c) = plt.subplots(2, 1, figsize=(5.3, 6.2), sharex=True)
    for ax in (ax_e, ax_c):
        ax.set_xscale("log")
        ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.1)
        ax.tick_params(axis="both", which="minor", direction="out", length=3, width=1.0)

    ax_e.errorbar(
        e0["Rigidity"],
        e0["Min_Etot"],
        yerr=[np.zeros(len(e0)), e0["Min_Etot_err_u"]],
        fmt="o",
        color="black",
        ecolor="0.2",
        capsize=5.5,
        capthick=1.6,
        label="0",
    )
    ax_e.errorbar(
        e0866["Rigidity"],
        e0866["Min_Etot"],
        yerr=[np.zeros(len(e0866)), e0866["Min_Etot_err_u"]],
        fmt="s",
        color="0.45",
        ecolor="0.45",
        capsize=5.5,
        capthick=1.6,
        label="0.866",
    )
    ax_e.axhline(216.5, color="0.25", linestyle="-.", linewidth=1.0)
    ax_e.axhline(214.2, color="0.45", linestyle="--", linewidth=1.0)
    label_box = {"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.6}
    ax_e.text(9.0, 217.0, "simulation min (rigid body)", fontsize=6, va="bottom", color="0.25", bbox=label_box)
    ax_e.text(9.0, 213.6, "theoretical min (rigid body)", fontsize=6, va="top", color="0.35", bbox=label_box)
    ax_e.set_ylabel(r"Minimum Total Energy (pN$\cdot$nm)")
    ax_e.set_ylim(88, 224)
    legend = ax_e.legend(title="Eccentricity", frameon=True, fancybox=False, loc="center right")
    legend.get_frame().set_edgecolor("0.35")

    ax_c.errorbar(
        e0["Rigidity"],
        e0["MeanCurvatureCap"],
        yerr=e0["MeanCurvatureCap_se"],
        fmt="o",
        color="black",
        ecolor="0.2",
        capsize=5.5,
        capthick=1.6,
    )
    ax_c.errorbar(
        e0866["Rigidity"],
        e0866["MeanCurvatureCap"],
        yerr=e0866["MeanCurvatureCap_se"],
        fmt="s",
        color="0.45",
        ecolor="0.45",
        capsize=5.5,
        capthick=1.6,
    )
    ax_c.set_ylabel(r"Mean Curvature of Cap (nm$^{-1}$)")
    ax_c.set_xlabel(r"Protein Lattice Rigidity (pN$\cdot$nm or pN/nm)")
    ax_c.set_xlim(8, 1200)
    fig.align_ylabels([ax_e, ax_c])
    fig.tight_layout()
    save(fig, "a30b30_a30b15_rigidity_energy_curvature")


def plot_cylindrical_asymmetry() -> None:
    summary = pd.read_csv(
        DATA
        / "curvature_asymmetry"
        / "cylindrical_symmetry_surface_summary_iqr_outliers_removed.csv"
    )
    fig, ax = plt.subplots(figsize=(5.7, 3.9))
    styles = {
        0.0: {"fmt": "o", "color": "black", "label": "0"},
        0.866: {"fmt": "s", "color": "0.45", "label": "0.866"},
    }
    for ecc, style in styles.items():
        subset = summary[np.isclose(summary["Ecc"], ecc)].sort_values("Rigidity")
        ax.errorbar(
            subset["Rigidity"],
            subset["CylindricalAsymmetry"],
            yerr=subset["StdErrCylindricalAsymmetry"],
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
    fig.tight_layout()
    save(fig, "cylindrical_symmetry_vs_rigidity_iqr_outliers_removed")


def cap_mask(vertices: np.ndarray, radius: float = 30.0) -> np.ndarray:
    return np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2) <= radius


def face_values(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.nanmean(values[faces], axis=1)


def plot_mean_curvature_heatmaps() -> None:
    surface_dir = DATA / "selected_minimum_surfaces"
    items = []
    for path in sorted(surface_dir.glob("*_selected_surface.npz")):
        with np.load(path) as data:
            items.append(
                {
                    "ecc": float(data["ecc"]),
                    "rigidity": float(data["rigidity"]),
                    "vertices": data["vertices"],
                    "faces": data["faces"],
                    "mean_curvature": data["mean_curvature"],
                }
            )
    values = np.concatenate(
        [item["mean_curvature"][cap_mask(item["vertices"]) & np.isfinite(item["mean_curvature"])] for item in items]
    )
    norm = mpl.colors.Normalize(vmin=0.0, vmax=float(np.nanpercentile(values, 98)))
    cmap = mpl.cm.viridis
    fig, axes = plt.subplots(2, 5, figsize=(14.2, 5.2), constrained_layout=False)
    for row_index, ecc in enumerate([0.0, 0.866]):
        for col_index, rigidity in enumerate(RIGIDITIES):
            ax = axes[row_index, col_index]
            matches = [
                item
                for item in items
                if np.isclose(item["ecc"], ecc) and np.isclose(item["rigidity"], rigidity)
            ]
            if not matches:
                ax.axis("off")
                continue
            item = matches[0]
            vertices = item["vertices"]
            faces = item["faces"]
            mean_curv = item["mean_curvature"]
            face_cap = np.any(cap_mask(vertices)[faces], axis=1)
            faces_to_plot = faces[face_cap]
            collection = PolyCollection(
                [vertices[tri, :2] for tri in faces_to_plot],
                array=face_values(mean_curv, faces_to_plot),
                cmap=cmap,
                norm=norm,
                linewidths=0.08,
                edgecolors=(1, 1, 1, 0.20),
                rasterized=True,
            )
            ax.add_collection(collection)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-36, 36)
            ax.set_ylim(-36, 36)
            ax.axis("off")
            if row_index == 0:
                ax.set_title(f"K = {rigidity:g}", fontsize=13, pad=3)
            if col_index == 0:
                ax.text(-0.08, 0.5, f"e = {ecc:g}", transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=13)
    cax = fig.add_axes([0.925, 0.18, 0.012, 0.64])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(r"Mean curvature magnitude (nm$^{-1}$)", fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.04, right=0.91, top=0.92, bottom=0.04, wspace=0.04, hspace=0.04)
    save(fig, "mean_curvature_heatmaps")


def plot_hoomd_curvature_vs_ngags() -> None:
    data = pd.read_csv(DATA / "hoomd_hiv_gag_geometry" / "hoomd_mean_cap_curvature_vs_ngags.csv")
    styles = {
        "k = 250": {"fmt": "o-", "color": "#2f7f73", "label": "deformable"},
        "rigid": {"fmt": "s-", "color": "#5b5b5b", "label": "rigid"},
    }
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for condition, style in styles.items():
        subset = data[data["Condition"].eq(condition)].sort_values("Ngags")
        ax.errorbar(
            subset["Ngags"],
            subset["MeanAbsCurvatureCapPerNm"],
            yerr=subset["SEPerNm"],
            fmt=style["fmt"],
            color=style["color"],
            ecolor=style["color"],
            capsize=5,
            capthick=1.5,
            markersize=6,
            label=style["label"],
        )
    ax.set_xlabel("Number of Gags")
    ax.set_ylabel(r"Mean Curvature of Cap (nm$^{-1}$)")
    legend = ax.legend(title="Protein lattice", frameon=True)
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    save(fig, "hoomd_mean_cap_curvature_vs_ngags")


def main() -> None:
    setup_style()
    plot_rigidity_energy_curvature()
    plot_cylindrical_asymmetry()
    plot_mean_curvature_heatmaps()
    plot_hoomd_curvature_vs_ngags()
    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
