from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from plot_rigidity_side_views import (
    ALLOWED_RIGIDITIES,
    PLOTS,
    RESULTS,
    center_vertices,
    read_faces,
    read_xyz_csv,
    select_a30b15,
    select_a30b30,
)
from plot_cylindrical_symmetry_rigidity import candidate_rows


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


def cotangent(u: np.ndarray, v: np.ndarray) -> float:
    cross_norm = np.linalg.norm(np.cross(u, v))
    if cross_norm < 1e-12:
        return 0.0
    return float(np.dot(u, v) / cross_norm)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-12:
        return 0.0
    value = np.clip(np.dot(u, v) / denom, -1.0, 1.0)
    return float(np.arccos(value))


def boundary_vertices(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    edge_counts: dict[tuple[int, int], int] = {}
    for i, j, k in faces:
        for a, b in ((i, j), (j, k), (k, i)):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = np.zeros(n_vertices, dtype=bool)
    for (i, j), count in edge_counts.items():
        if count == 1:
            boundary[i] = True
            boundary[j] = True
    return boundary


def curvature_difference(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate kappa_max - kappa_min from cotangent mean curvature and angle-deficit Gaussian curvature."""
    n_vertices = len(vertices)
    lap = np.zeros((n_vertices, 3), dtype=float)
    mixed_area = np.zeros(n_vertices, dtype=float)
    angle_sum = np.zeros(n_vertices, dtype=float)

    for tri in faces:
        i, j, k = map(int, tri)
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        face_area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
        if face_area <= 1e-12:
            continue

        mixed_area[[i, j, k]] += face_area / 3.0
        angle_i = angle_between(vj - vi, vk - vi)
        angle_j = angle_between(vi - vj, vk - vj)
        angle_k = angle_between(vi - vk, vj - vk)
        angle_sum[i] += angle_i
        angle_sum[j] += angle_j
        angle_sum[k] += angle_k

        cot_i = cotangent(vj - vi, vk - vi)
        cot_j = cotangent(vi - vj, vk - vj)
        cot_k = cotangent(vi - vk, vj - vk)
        lap[i] += cot_k * (vj - vi) + cot_j * (vk - vi)
        lap[j] += cot_k * (vi - vj) + cot_i * (vk - vj)
        lap[k] += cot_j * (vi - vk) + cot_i * (vj - vk)

    valid = mixed_area > 1e-12
    mean_curv = np.full(n_vertices, np.nan, dtype=float)
    gaussian_curv = np.full(n_vertices, np.nan, dtype=float)
    mean_curv[valid] = 0.5 * np.linalg.norm(lap[valid] / (2.0 * mixed_area[valid, None]), axis=1)

    boundary = boundary_vertices(faces, n_vertices)
    interior = valid & ~boundary
    gaussian_curv[interior] = (2.0 * np.pi - angle_sum[interior]) / mixed_area[interior]

    discriminant = mean_curv**2 - gaussian_curv
    discriminant = np.where(np.isfinite(discriminant), np.maximum(discriminant, 0.0), np.nan)
    kappa_diff = 2.0 * np.sqrt(discriminant)
    return mean_curv, gaussian_curv, kappa_diff


def cap_mask(vertices: np.ndarray, radius: float = 30.0) -> np.ndarray:
    return np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2) <= radius


def face_values(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.nanmean(values[faces], axis=1)


def topdown_polygons(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    return [vertices[tri, :2] for tri in faces]


def load_selected() -> pd.DataFrame:
    selected = pd.concat([select_a30b30(), select_a30b15()], ignore_index=True)
    selected = selected.sort_values(["Ecc", "Rigidity"]).reset_index(drop=True)
    return selected


def kappa_diff_for_row(row: pd.Series) -> tuple[float, float, int, dict]:
    vertices = read_xyz_csv(Path(row["VertexFile"]))
    faces = read_faces(Path(row["FaceFile"]))
    vertices, _ = center_vertices(vertices, np.empty((0, 3), dtype=float))
    mean_curv, _, kappa_diff = curvature_difference(vertices, faces)
    mask = cap_mask(vertices, radius=30.0) & np.isfinite(kappa_diff)
    values = kappa_diff[mask]
    mean = float(np.nanmean(values)) if len(values) else np.nan
    within_mesh_stderr = float(np.nanstd(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    map_item = {
        "row": row,
        "vertices": vertices,
        "faces": faces,
        "mean_curv": mean_curv,
        "kappa_diff": kappa_diff,
        "mask": mask,
    }
    return mean, within_mesh_stderr, int(len(values)), map_item


def bootstrap_joint_minimum(candidates: pd.DataFrame, n_bootstrap: int = 10000) -> pd.DataFrame:
    rows = []
    for (ecc, rigidity), subset in candidates.groupby(["Ecc", "Rigidity"], dropna=False):
        subset = subset[np.isfinite(subset["MeanKappaDiffCap"])].reset_index(drop=True)
        if subset.empty:
            continue
        observed = subset.loc[subset["Min_Etot"].idxmin()].copy()
        values = subset[["Min_Etot", "MeanKappaDiffCap"]].to_numpy(float)
        seed = 20260616 + int(round(1000 * float(ecc))) + int(round(float(rigidity)))
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        resampled = values[sample_indices]
        min_indices = np.argmin(resampled[:, :, 0], axis=1)
        selected_values = resampled[np.arange(n_bootstrap), min_indices, 1]
        rows.append(
            {
                "Ecc": float(ecc),
                "Rigidity": float(rigidity),
                "MeanKappaDiffCap": float(observed["MeanKappaDiffCap"]),
                "StdErrKappaDiffCap": float(np.std(selected_values, ddof=1)),
                "BootstrapKappaDiffP025": float(np.percentile(selected_values, 2.5)),
                "BootstrapKappaDiffP975": float(np.percentile(selected_values, 97.5)),
                "WithinMeshStdErrKappaDiffCap": float(observed["WithinMeshStdErrKappaDiffCap"]),
                "Runs": int(len(subset)),
                "NCapVertices": int(observed["NCapVertices"]),
                "RunID": int(observed["RunID"]),
                "RelaxArea": float(observed["RelaxArea"]),
                "VertexFile": observed["VertexFile"],
                "FaceFile": observed["FaceFile"],
                "Min_Etot": float(observed["Min_Etot"]),
                "Min_step": float(observed["Min_step"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Ecc", "Rigidity"]).reset_index(drop=True)


def compute_summary(selected: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    summary_rows = []
    maps = []
    candidates = candidate_rows()
    for _, row in candidates.iterrows():
        mean, within_mesh_stderr, n_cap_vertices, _ = kappa_diff_for_row(row)
        summary_rows.append(
            {
                "Ecc": float(row["Ecc"]),
                "Rigidity": float(row["Rigidity"]),
                "RunID": int(row["RunID"]),
                "RelaxArea": float(row["RelaxArea"]),
                "MeanKappaDiffCap": mean,
                "WithinMeshStdErrKappaDiffCap": within_mesh_stderr,
                "NCapVertices": n_cap_vertices,
                "VertexFile": row["VertexFile"],
                "FaceFile": row["FaceFile"],
                "Min_Etot": row["Min_Etot"],
                "Min_step": row["Min_step"],
            }
        )
    candidates_with_metrics = pd.DataFrame(summary_rows)
    candidates_with_metrics.to_csv(RESULTS / "curvature_difference_candidate_trajectories.csv", index=False)
    summary = bootstrap_joint_minimum(candidates_with_metrics)
    summary.to_csv(RESULTS / "curvature_difference_cap_summary.csv", index=False)

    for _, row in summary.iterrows():
        _, _, _, map_item = kappa_diff_for_row(row)
        maps.append(map_item)
    return summary, maps


def plot_heatmaps(maps: list[dict]) -> None:
    all_values = np.concatenate([item["kappa_diff"][item["mask"]] for item in maps])
    vmax = float(np.nanpercentile(all_values, 98))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = mpl.cm.viridis

    fig, axes = plt.subplots(2, 5, figsize=(14.2, 5.2), constrained_layout=False)
    for row_index, ecc in enumerate([0.0, 0.866]):
        for col_index, rigidity in enumerate(ALLOWED_RIGIDITIES):
            ax = axes[row_index, col_index]
            matches = [
                item
                for item in maps
                if np.isclose(float(item["row"]["Ecc"]), ecc)
                and np.isclose(float(item["row"]["Rigidity"]), rigidity)
            ]
            if not matches:
                ax.axis("off")
                continue
            item = matches[0]
            vertices = item["vertices"]
            faces = item["faces"]
            values = item["kappa_diff"]
            cap = cap_mask(vertices, radius=30.0)
            face_cap = np.any(cap[faces], axis=1)
            faces_to_plot = faces[face_cap]
            polygons = topdown_polygons(vertices, faces_to_plot)
            colors = face_values(values, faces_to_plot)
            collection = PolyCollection(
                polygons,
                array=colors,
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
                ax.text(
                    -0.08,
                    0.5,
                    f"e = {ecc:g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=13,
                )

    cax = fig.add_axes([0.925, 0.18, 0.012, 0.64])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(r"$\kappa_{\max} - \kappa_{\min}$ (nm$^{-1}$)", fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.04, right=0.91, top=0.92, bottom=0.04, wspace=0.04, hspace=0.04)
    for suffix in ("png", "svg"):
        out = PLOTS / f"curvature_difference_heatmaps.{suffix}"
        fig.savefig(out, dpi=450 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.02)
        print(out)


def plot_mean_curvature_heatmaps(maps: list[dict]) -> None:
    all_values = np.concatenate([item["mean_curv"][item["mask"]] for item in maps])
    vmax = float(np.nanpercentile(all_values, 98))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = mpl.cm.viridis

    fig, axes = plt.subplots(2, 5, figsize=(14.2, 5.2), constrained_layout=False)
    for row_index, ecc in enumerate([0.0, 0.866]):
        for col_index, rigidity in enumerate(ALLOWED_RIGIDITIES):
            ax = axes[row_index, col_index]
            matches = [
                item
                for item in maps
                if np.isclose(float(item["row"]["Ecc"]), ecc)
                and np.isclose(float(item["row"]["Rigidity"]), rigidity)
            ]
            if not matches:
                ax.axis("off")
                continue
            item = matches[0]
            vertices = item["vertices"]
            faces = item["faces"]
            values = item["mean_curv"]
            cap = cap_mask(vertices, radius=30.0)
            face_cap = np.any(cap[faces], axis=1)
            faces_to_plot = faces[face_cap]
            polygons = topdown_polygons(vertices, faces_to_plot)
            colors = face_values(values, faces_to_plot)
            collection = PolyCollection(
                polygons,
                array=colors,
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
                ax.text(
                    -0.08,
                    0.5,
                    f"e = {ecc:g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=13,
                )

    cax = fig.add_axes([0.925, 0.18, 0.012, 0.64])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(r"Mean curvature magnitude (nm$^{-1}$)", fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.04, right=0.91, top=0.92, bottom=0.04, wspace=0.04, hspace=0.04)
    for suffix in ("png", "svg"):
        out = PLOTS / f"mean_curvature_heatmaps.{suffix}"
        fig.savefig(out, dpi=450 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.02)
        print(out)


def plot_rigidity(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 3.9))
    styles = {
        0.0: {"fmt": "o", "color": "black", "label": "0"},
        0.866: {"fmt": "s", "color": "0.45", "label": "0.866"},
    }
    for ecc, style in styles.items():
        data = summary[np.isclose(summary["Ecc"], ecc)].sort_values("Rigidity")
        ax.errorbar(
            data["Rigidity"],
            data["MeanKappaDiffCap"],
            yerr=data["StdErrKappaDiffCap"],
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
    ax.set_ylabel(r"Mean $\kappa_{\max} - \kappa_{\min}$ (nm$^{-1}$)")
    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.1)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=1.0)
    legend = ax.legend(title="Eccentricity", frameon=True, fancybox=False, loc="best")
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        out = PLOTS / f"curvature_difference_vs_rigidity.{suffix}"
        fig.savefig(out, dpi=450 if suffix == "png" else None, bbox_inches="tight")
        print(out)


def main() -> None:
    selected = load_selected()
    selected.to_csv(RESULTS / "curvature_difference_selected_minima.csv", index=False)
    summary, maps = compute_summary(selected)
    plot_heatmaps(maps)
    plot_mean_curvature_heatmaps(maps)
    plot_rigidity(summary)


if __name__ == "__main__":
    main()
