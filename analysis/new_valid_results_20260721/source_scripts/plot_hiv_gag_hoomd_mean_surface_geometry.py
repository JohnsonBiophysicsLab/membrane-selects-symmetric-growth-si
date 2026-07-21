#!/usr/bin/env python3
"""Plot second-half mean HOOMD membrane geometries.

The input `*_canonical_surface_arrays.npz` files contain `z_mean`, where each
membrane node has already been averaged over the second half of the simulation.
This script intentionally plots only the membrane mean surface, so it does not
need GSD-derived Gag COM positions.
"""

from __future__ import annotations

import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_hiv_gag_hoomd_geometry import (
    CONDITIONS,
    K250_DIR,
    PLOT_DIR,
    RESULT_DIR,
    RIGID_DIR,
    SYSTEMS,
    load_surface,
    oriented_cap_surface,
    read_summary,
    set_equal_3d_scale,
    setup_style,
    uniform_window_crop,
    uniform_window_half_width,
)


def load_inputs() -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, dict[str, dict[str, np.ndarray]]]]:
    summary = {
        "k250": read_summary(K250_DIR / "canonical_surface_summary.csv"),
        "rigid": read_summary(RIGID_DIR / "canonical_surface_summary.csv"),
    }
    surfaces: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for cond_key, _, folder, _, _ in CONDITIONS:
        surfaces[cond_key] = {system: load_surface(folder, system) for system in SYSTEMS}
    return summary, surfaces


def oriented_surfaces(
    surfaces: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, dict[str, np.ndarray | float]]]:
    return {
        cond_key: {system: oriented_cap_surface(surfaces[cond_key][system]) for system in SYSTEMS}
        for cond_key, *_rest in CONDITIONS
    }


def plot_mean_side_views(oriented: dict[str, dict[str, dict[str, np.ndarray | float]]]) -> None:
    half_width = uniform_window_half_width(oriented)
    all_z = []
    for cond_key, *_rest in CONDITIONS:
        for system in SYSTEMS:
            arrays = oriented[cond_key][system]
            crop = uniform_window_crop(arrays, half_width)
            all_z.append(np.asarray(arrays["z"])[crop])
    z_min = min(float(np.nanmin(z)) for z in all_z)
    z_max = max(float(np.nanmax(z)) for z in all_z)
    norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(13.2, 5.3))
    for row, (cond_key, cond_label, *_rest) in enumerate(CONDITIONS):
        for col, system in enumerate(SYSTEMS):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
            arrays = oriented[cond_key][system]
            crop = uniform_window_crop(arrays, half_width)
            x = np.asarray(arrays["x"])[crop]
            y = np.asarray(arrays["y"])[crop]
            z = np.asarray(arrays["z"])[crop]
            ax.plot_surface(
                x,
                y,
                z,
                rstride=1,
                cstride=1,
                facecolors=cmap(norm(z)),
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=0.96,
            )
            ax.view_init(elev=4, azim=-90)
            set_equal_3d_scale(ax, x, y, z)
            ax.set_axis_off()
            ax.set_proj_type("ortho")
            if row == 0:
                ax.set_title(f"{system} Gags", fontsize=15, pad=-4)
            ax.text2D(-0.03, 0.5, cond_label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=15)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03, wspace=-0.12, hspace=-0.2)
    for ext in ("png", "svg"):
        fig.savefig(PLOT_DIR / f"hoomd_mean_second_half_side_views_equal_scale.{ext}", dpi=350)
    plt.close(fig)


def plot_mean_topdown_heatmaps(oriented: dict[str, dict[str, dict[str, np.ndarray | float]]]) -> None:
    half_width = uniform_window_half_width(oriented)
    all_z = []
    for cond_key, *_rest in CONDITIONS:
        for system in SYSTEMS:
            arrays = oriented[cond_key][system]
            crop = uniform_window_crop(arrays, half_width)
            all_z.append(np.asarray(arrays["z"])[crop])
    z_abs = max(max(abs(float(np.nanmin(z))), abs(float(np.nanmax(z)))) for z in all_z)
    z_min = -z_abs
    z_max = z_abs

    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.0), constrained_layout=True)
    mesh = None
    for row, (cond_key, cond_label, *_rest) in enumerate(CONDITIONS):
        for col, system in enumerate(SYSTEMS):
            ax = axes[row, col]
            arrays = oriented[cond_key][system]
            crop = uniform_window_crop(arrays, half_width)
            x = np.asarray(arrays["x"])[crop]
            y = np.asarray(arrays["y"])[crop]
            z = np.asarray(arrays["z"])[crop]
            mesh = ax.pcolormesh(x, y, z, cmap="viridis", shading="nearest", vmin=z_min, vmax=z_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-half_width, half_width)
            ax.set_ylim(-half_width, half_width)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
            if row == 0:
                ax.set_title(f"{system} Gags", fontsize=15)
            if col == 0:
                ax.set_ylabel(cond_label, fontsize=15)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.82, pad=0.012)
    cbar.set_label("Second-half Mean Oriented Height (nm)", fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    for ext in ("png", "svg"):
        fig.savefig(PLOT_DIR / f"hoomd_mean_second_half_topdown_height_heatmaps.{ext}", dpi=350)
    plt.close(fig)


def write_metadata(summary: dict[str, dict[str, dict[str, str]]]) -> None:
    path = RESULT_DIR / "hoomd_mean_second_half_geometry_plot_metadata.csv"
    fieldnames = ["ConditionKey", "Condition", "System", "CanonicalFrames", "NodeStatistic", "SourceDirectory"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cond_key, cond_label, folder, *_rest in CONDITIONS:
            for system in SYSTEMS:
                writer.writerow(
                    {
                        "ConditionKey": cond_key,
                        "Condition": cond_label,
                        "System": system,
                        "CanonicalFrames": summary[cond_key][system]["CanonicalFrames"],
                        "NodeStatistic": "z_mean, each membrane node averaged over trajectory second half",
                        "SourceDirectory": str(folder),
                    }
                )


def main() -> None:
    setup_style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary, surfaces = load_inputs()
    oriented = oriented_surfaces(surfaces)
    plot_mean_side_views(oriented)
    plot_mean_topdown_heatmaps(oriented)
    write_metadata(summary)
    print(f"Wrote mean second-half geometry plots to {PLOT_DIR}")
    print(f"Wrote metadata to {RESULT_DIR / 'hoomd_mean_second_half_geometry_plot_metadata.csv'}")


if __name__ == "__main__":
    main()
