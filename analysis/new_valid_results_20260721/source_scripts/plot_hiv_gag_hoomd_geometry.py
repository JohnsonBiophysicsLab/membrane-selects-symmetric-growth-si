#!/usr/bin/env python3
"""Plot corrected HOOMD HIV Gag geometry comparisons.

Inputs are the refreshed `canonical_surface_summary.csv` and
`*_canonical_surface_arrays.npz` files produced by
`analysis/hoomd_k250_geometry/analyze_hoomd_k250_geometry.py`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
K250_DIR = ROOT / "analysis" / "hoomd_k250_geometry" / "remote_results"
RIGID_DIR = ROOT / "analysis" / "hoomd_rigid_geometry" / "remote_results"
OUT_DIR = ROOT / "analysis" / "hoomd_geometry_plots"
PLOT_DIR = OUT_DIR / "plots"
RESULT_DIR = OUT_DIR / "results"
SYSTEMS = ["203", "252", "408", "454"]
GAG_POSITION_CSV = RESULT_DIR / "hoomd_canonical_gag_com_positions.csv"
CONDITIONS = [
    ("k250", "k = 250", K250_DIR, "#2f7f73", "o"),
    ("rigid", "rigid", RIGID_DIR, "#5b5b5b", "s"),
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "svg.fonttype": "none",
        }
    )


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {row["System"]: row for row in rows}


def load_surface(folder: Path, system: str) -> dict[str, np.ndarray]:
    with np.load(folder / f"{system}_canonical_surface_arrays.npz") as data:
        return {key: data[key] for key in data.files}


def cap_crop(arrays: dict[str, np.ndarray], margin_nm: float = 25.0) -> tuple[slice, slice]:
    x = arrays["x"]
    y = arrays["y"]
    cap = arrays["cap_mask"].astype(bool)
    if not np.any(cap):
        return slice(None), slice(None)
    dx = float(np.nanmedian(np.diff(x[:, 0])))
    dy = float(np.nanmedian(np.diff(y[0, :])))
    pad_x = max(2, int(np.ceil(margin_nm / abs(dx))))
    pad_y = max(2, int(np.ceil(margin_nm / abs(dy))))
    ix, iy = np.where(cap)
    x0 = max(0, int(ix.min()) - pad_x)
    x1 = min(x.shape[0], int(ix.max()) + pad_x + 1)
    y0 = max(0, int(iy.min()) - pad_y)
    y1 = min(y.shape[1], int(iy.max()) + pad_y + 1)
    return slice(x0, x1), slice(y0, y1)


def uniform_window_half_width(oriented: dict[str, dict[str, dict[str, np.ndarray | float]]], margin_nm: float = 25.0) -> float:
    half_width = 0.0
    for cond_key in oriented:
        for system in oriented[cond_key]:
            arrays = oriented[cond_key][system]
            x = np.asarray(arrays["x"], dtype=float)
            y = np.asarray(arrays["y"], dtype=float)
            cap = np.asarray(arrays["cap_mask"], dtype=bool)
            if np.any(cap):
                half_width = max(half_width, float(np.nanmax(np.abs(x[cap]))), float(np.nanmax(np.abs(y[cap]))))
    return max(half_width + margin_nm, 45.0)


def uniform_window_crop(arrays: dict[str, np.ndarray | float], half_width: float) -> tuple[slice, slice]:
    x = np.asarray(arrays["x"], dtype=float)
    y = np.asarray(arrays["y"], dtype=float)
    x_axis = x[:, 0]
    y_axis = y[0, :]
    x_indices = np.where((x_axis >= -half_width) & (x_axis <= half_width))[0]
    y_indices = np.where((y_axis >= -half_width) & (y_axis <= half_width))[0]
    if len(x_indices) == 0 or len(y_indices) == 0:
        return slice(None), slice(None)
    return slice(int(x_indices[0]), int(x_indices[-1]) + 1), slice(int(y_indices[0]), int(y_indices[-1]) + 1)


def oriented_cap_surface(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    """Return a recentered plotting surface with the cap shown concave down.

    The numerical analysis is unchanged. This is a visualization transform:
    x/y are shifted so the attached-node cap centroid is at the origin, z is
    referenced to a local non-cap annulus, and z is flipped only when needed so
    the attached cap sits above that local reference in the plotted view.
    """
    x = np.asarray(arrays["x"], dtype=float)
    y = np.asarray(arrays["y"], dtype=float)
    z = np.asarray(arrays["z_mean"], dtype=float)
    cap = np.asarray(arrays["cap_mask"], dtype=bool)
    if not np.any(cap):
        return {
            "x": x,
            "y": y,
            "z": z,
            "cap_mask": cap,
            "cap_attachment_multiplicity": np.asarray(arrays.get("cap_attachment_multiplicity", cap.astype(int))),
            "sign": 1.0,
            "reference_z": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }

    weights = np.asarray(arrays.get("cap_attachment_frequency", cap.astype(float)), dtype=float)
    weights = np.where(cap & np.isfinite(weights) & (weights > 0), weights, 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = cap.astype(float)
    center_x = float(np.sum(x * weights) / np.sum(weights))
    center_y = float(np.sum(y * weights) / np.sum(weights))
    x0 = x - center_x
    y0 = y - center_y

    r = np.sqrt(x0 * x0 + y0 * y0)
    cap_radius = float(np.nanquantile(r[cap], 0.95))
    annulus = (~cap) & (r >= cap_radius + 8.0) & (r <= cap_radius + 28.0)
    if int(np.sum(annulus)) < 20:
        annulus = ~cap
    reference_z = float(np.nanmean(z[annulus]))
    z0 = z - reference_z
    cap_mean = float(np.nanmean(z0[cap]))
    annulus_mean = float(np.nanmean(z0[annulus]))
    sign = 1.0 if cap_mean >= annulus_mean else -1.0
    return {
        "x": x0,
        "y": y0,
        "z": sign * z0,
        "cap_mask": cap,
        "cap_attachment_multiplicity": np.asarray(arrays.get("cap_attachment_multiplicity", cap.astype(int))),
        "sign": sign,
        "reference_z": reference_z,
        "center_x": center_x,
        "center_y": center_y,
    }


def load_true_gag_positions() -> dict[tuple[str, str], np.ndarray]:
    if not GAG_POSITION_CSV.exists():
        raise FileNotFoundError(
            f"{GAG_POSITION_CSV} is required for geometry plots. "
            "Run extract_hoomd_gag_com_positions.py on the Rockfish folders that contain trajectory.gsd, "
            "then rsync the resulting CSV into this results directory. "
            "The plotter intentionally refuses to draw guessed Gag positions."
        )
    output: dict[tuple[str, str], list[list[float]]] = {}
    with GAG_POSITION_CSV.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["ConditionKey"], row["System"])
            output.setdefault(key, []).append(
                [
                    float(row["XMeanNm"]),
                    float(row["YMeanNm"]),
                    float(row["ZMeanNm"]),
                ]
            )
    positions = {key: np.asarray(value, dtype=float) for key, value in output.items()}
    required = {(cond_key, system) for cond_key, *_rest in CONDITIONS for system in SYSTEMS}
    missing = sorted(required.difference(positions))
    if missing:
        formatted = ", ".join(f"{cond}:{system}" for cond, system in missing)
        raise ValueError(f"{GAG_POSITION_CSV} is missing GSD-derived Gag COM rows for: {formatted}")
    return positions


def nearest_surface_z(arrays: dict[str, np.ndarray | float], px: np.ndarray, py: np.ndarray) -> np.ndarray:
    x_axis = np.asarray(arrays["x"], dtype=float)[:, 0]
    y_axis = np.asarray(arrays["y"], dtype=float)[0, :]
    z = np.asarray(arrays["z"], dtype=float)
    ix = np.searchsorted(x_axis, px)
    iy = np.searchsorted(y_axis, py)
    ix = np.clip(ix, 1, len(x_axis) - 1)
    iy = np.clip(iy, 1, len(y_axis) - 1)
    ix = np.where(np.abs(px - x_axis[ix - 1]) <= np.abs(px - x_axis[ix]), ix - 1, ix)
    iy = np.where(np.abs(py - y_axis[iy - 1]) <= np.abs(py - y_axis[iy]), iy - 1, iy)
    return z[ix, iy]


def overlay_gag_points(
    arrays: dict[str, np.ndarray | float],
    cond_key: str,
    system: str,
    true_gag_positions: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    true_positions = true_gag_positions.get((cond_key, system))
    if true_positions is None or len(true_positions) == 0:
        raise ValueError(f"Missing GSD-derived Gag COM positions for {cond_key}:{system}")
    sign = float(arrays["sign"])
    reference_z = float(arrays["reference_z"])
    center_x = float(arrays["center_x"])
    center_y = float(arrays["center_y"])
    return (
        true_positions[:, 0] - center_x,
        true_positions[:, 1] - center_y,
        sign * (true_positions[:, 2] - reference_z),
        "gsd_gag_com",
    )


def gag_side_summary(
    arrays: dict[str, np.ndarray | float],
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    half_width: float,
) -> dict[str, float]:
    in_window = (gx >= -half_width) & (gx <= half_width) & (gy >= -half_width) & (gy <= half_width)
    if not np.any(in_window):
        return {"MedianGagMinusMembraneZNm": float("nan"), "FractionGagBelowMembrane": float("nan")}
    surface_z = nearest_surface_z(arrays, gx[in_window], gy[in_window])
    delta = gz[in_window] - surface_z
    return {
        "MedianGagMinusMembraneZNm": float(np.nanmedian(delta)),
        "FractionGagBelowMembrane": float(np.mean(delta < 0.0)),
    }


def set_equal_3d_scale(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    half = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_side_views(summary: dict[str, dict[str, dict[str, str]]], surfaces: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    all_z = []
    oriented = {}
    true_gag_positions = load_true_gag_positions()
    for cond_key, _, _, _, _ in CONDITIONS:
        oriented[cond_key] = {}
        for system in SYSTEMS:
            oriented[cond_key][system] = oriented_cap_surface(surfaces[cond_key][system])
    half_width = uniform_window_half_width(oriented)
    for cond_key, _, _, _, _ in CONDITIONS:
        for system in SYSTEMS:
            z = oriented[cond_key][system]["z"]
            crop = uniform_window_crop(oriented[cond_key][system], half_width)
            all_z.append(z[crop])
    z_min = min(float(np.nanmin(z)) for z in all_z)
    z_max = max(float(np.nanmax(z)) for z in all_z)
    norm = mpl.colors.Normalize(vmin=z_min, vmax=z_max)
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(13.2, 5.3))
    axes = []
    for row, (cond_key, cond_label, _, _, _) in enumerate(CONDITIONS):
        for col, system in enumerate(SYSTEMS):
            ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection="3d")
            arrays = oriented[cond_key][system]
            crop = uniform_window_crop(arrays, half_width)
            x = arrays["x"][crop]
            y = arrays["y"][crop]
            z = arrays["z"][crop]
            facecolors = cmap(norm(z))
            ax.plot_surface(
                x,
                y,
                z,
                rstride=1,
                cstride=1,
                facecolors=facecolors,
                linewidth=0,
                antialiased=True,
                shade=False,
                alpha=0.95,
            )
            gx, gy, gz, _gag_mode = overlay_gag_points(arrays, cond_key, system, true_gag_positions)
            in_window = (gx >= -half_width) & (gx <= half_width) & (gy >= -half_width) & (gy <= half_width)
            if np.any(in_window):
                ax.scatter(
                    gx[in_window],
                    gy[in_window],
                    gz[in_window],
                    s=3.5,
                    c="black",
                    alpha=0.40,
                    depthshade=False,
                )
            ax.view_init(elev=4, azim=-90)
            set_equal_3d_scale(ax, x, y, z)
            ax.set_axis_off()
            ax.set_proj_type("ortho")
            if row == 0:
                ax.set_title(f"{system} Gags", fontsize=15, pad=-4)
            ax.text2D(-0.03, 0.5, cond_label, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=15)
            axes.append(ax)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03, wspace=-0.12, hspace=-0.2)
    for ext in ("png", "svg"):
        fig.savefig(PLOT_DIR / f"hoomd_gag_side_views_equal_scale.{ext}", dpi=350)
    plt.close(fig)


def plot_heatmaps(surfaces: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    all_z = []
    oriented = {}
    true_gag_positions = load_true_gag_positions()
    for cond_key, _, _, _, _ in CONDITIONS:
        oriented[cond_key] = {}
        for system in SYSTEMS:
            oriented[cond_key][system] = oriented_cap_surface(surfaces[cond_key][system])
    half_width = uniform_window_half_width(oriented)
    for cond_key, _, _, _, _ in CONDITIONS:
        for system in SYSTEMS:
            arrays = oriented[cond_key][system]
            crop = uniform_window_crop(arrays, half_width)
            all_z.append(arrays["z"][crop])
    z_abs = max(max(abs(float(np.nanmin(z))), abs(float(np.nanmax(z)))) for z in all_z)
    z_min = -z_abs
    z_max = z_abs

    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.0), constrained_layout=True)
    mesh = None
    for row, (cond_key, cond_label, _, _, _) in enumerate(CONDITIONS):
        for col, system in enumerate(SYSTEMS):
            ax = axes[row, col]
            arrays = oriented[cond_key][system]
            crop = uniform_window_crop(arrays, half_width)
            x = arrays["x"][crop]
            y = arrays["y"][crop]
            z = arrays["z"][crop]
            mesh = ax.pcolormesh(x, y, z, cmap="viridis", shading="nearest", vmin=z_min, vmax=z_max)
            gx, gy, _gz, _gag_mode = overlay_gag_points(arrays, cond_key, system, true_gag_positions)
            in_window = (gx >= -half_width) & (gx <= half_width) & (gy >= -half_width) & (gy <= half_width)
            if np.any(in_window):
                ax.scatter(gx[in_window], gy[in_window], s=5, c="black", alpha=0.34, linewidths=0)
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
    cbar.set_label("Oriented Height (nm)", fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    for ext in ("png", "svg"):
        fig.savefig(PLOT_DIR / f"hoomd_gag_topdown_height_heatmaps.{ext}", dpi=350)
    plt.close(fig)


def write_curvature_csv(summary: dict[str, dict[str, dict[str, str]]]) -> None:
    out = RESULT_DIR / "hoomd_mean_cap_curvature_vs_ngags.csv"
    fieldnames = ["Condition", "Ngags", "MeanAbsCurvatureCapPerNm", "SEPerNm", "CapMaskMode", "CanonicalFrames"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cond_key, cond_label, _, _, _ in CONDITIONS:
            for system in SYSTEMS:
                row = summary[cond_key][system]
                writer.writerow(
                    {
                        "Condition": cond_label,
                        "Ngags": row["Ngags"],
                        "MeanAbsCurvatureCapPerNm": row["FrameMeanAbsCurvatureCapMeanPerNm"],
                        "SEPerNm": row["FrameMeanAbsCurvatureCapSemPerNm"],
                        "CapMaskMode": row["CapMaskMode"],
                        "CanonicalFrames": row["CanonicalFrames"],
                    }
                )


def write_geometry_plot_metadata(surfaces: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    oriented = {}
    true_gag_positions = load_true_gag_positions()
    for cond_key, _, _, _, _ in CONDITIONS:
        oriented[cond_key] = {system: oriented_cap_surface(surfaces[cond_key][system]) for system in SYSTEMS}
    half_width = uniform_window_half_width(oriented)
    rows = []
    for cond_key, cond_label, _, _, _ in CONDITIONS:
        for system in SYSTEMS:
            gx, gy, gz, mode = overlay_gag_points(oriented[cond_key][system], cond_key, system, true_gag_positions)
            side = gag_side_summary(oriented[cond_key][system], gx, gy, gz, half_width)
            rows.append(
                {
                    "ConditionKey": cond_key,
                    "Condition": cond_label,
                    "System": system,
                    "UniformHalfWidthNm": half_width,
                    "UniformFullWidthNm": 2.0 * half_width,
                    "GagOverlayMode": mode,
                    "GagOverlayPointCount": len(gx),
                    "GagMembraneOffsetNmIfFallback": "",
                    "MedianGagMinusMembraneZNm": side["MedianGagMinusMembraneZNm"],
                    "FractionGagBelowMembrane": side["FractionGagBelowMembrane"],
                }
            )
    path = RESULT_DIR / "hoomd_geometry_plot_metadata.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curvature_vs_ngags(summary: dict[str, dict[str, dict[str, str]]]) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.2))
    for cond_key, cond_label, _, color, marker in CONDITIONS:
        rows = [summary[cond_key][system] for system in SYSTEMS]
        x = np.asarray([float(row["Ngags"]) for row in rows])
        y = np.asarray([float(row["FrameMeanAbsCurvatureCapMeanPerNm"]) for row in rows])
        yerr = np.asarray([float(row["FrameMeanAbsCurvatureCapSemPerNm"]) for row in rows])
        order = np.argsort(x)
        ax.errorbar(
            x[order],
            y[order],
            yerr=yerr[order],
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7.5,
            capsize=4.5,
            capthick=1.3,
            elinewidth=1.3,
            linewidth=0,
            label=cond_label,
        )
        ax.plot(x[order], y[order], color=color, linewidth=1.4, alpha=0.85)
    ax.set_xlabel("Number of Gags")
    ax.set_ylabel("Mean Curvature of Cap (nm$^{-1}$)")
    ax.legend(frameon=True, title="Protein lattice", title_fontsize=14, loc="best")
    ax.grid(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(PLOT_DIR / f"hoomd_mean_cap_curvature_vs_ngags.{ext}", dpi=350)
    plt.close(fig)


def main() -> None:
    setup_style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "k250": read_summary(K250_DIR / "canonical_surface_summary.csv"),
        "rigid": read_summary(RIGID_DIR / "canonical_surface_summary.csv"),
    }
    surfaces: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for cond_key, _, folder, _, _ in CONDITIONS:
        surfaces[cond_key] = {system: load_surface(folder, system) for system in SYSTEMS}

    plot_side_views(summary, surfaces)
    plot_heatmaps(surfaces)
    write_curvature_csv(summary)
    write_geometry_plot_metadata(surfaces)
    plot_curvature_vs_ngags(summary)
    print(f"Wrote plots to {PLOT_DIR}")
    print(f"Wrote summary CSV to {RESULT_DIR / 'hoomd_mean_cap_curvature_vs_ngags.csv'}")


if __name__ == "__main__":
    main()
