from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_rigidity_side_views import (
    A30B15_BASE,
    A30B15_RUNS,
    A30B30_BASE,
    A30B30_RUNS,
    A30B15_PREFERRED_CONTINUATION_DATASET,
    ALLOWED_RIGIDITIES,
    PLOTS,
    RESULTS,
    center_vertices,
    load_a30b15_continuation_runs,
    nearest_vertex_file,
    normalize_rigidity,
    read_faces,
    read_xyz_csv,
)


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


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
        axis=1,
    )


def build_topology(faces: np.ndarray, n_vertices: int) -> tuple[dict[tuple[int, int], list[int]], list[set[int]]]:
    edge_opposites: dict[tuple[int, int], list[int]] = {}
    neighbors = [set() for _ in range(n_vertices)]
    for i, j, k in faces:
        i, j, k = int(i), int(j), int(k)
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            edge = tuple(sorted((a, b)))
            edge_opposites.setdefault(edge, []).append(c)
            neighbors[a].add(b)
            neighbors[b].add(a)
    return edge_opposites, neighbors


def loop_subdivide(vertices: np.ndarray, faces: np.ndarray, iterations: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the Loop subdivision limit surface with repeated refinement."""
    refined_vertices = vertices.astype(float).copy()
    refined_faces = faces.astype(int).copy()
    for _ in range(iterations):
        edge_opposites, neighbors = build_topology(refined_faces, len(refined_vertices))
        boundary_neighbors = [set() for _ in range(len(refined_vertices))]
        for (i, j), opposites in edge_opposites.items():
            if len(opposites) == 1:
                boundary_neighbors[i].add(j)
                boundary_neighbors[j].add(i)

        new_positions = np.empty_like(refined_vertices)
        for index, vertex in enumerate(refined_vertices):
            if boundary_neighbors[index]:
                b_neighbors = sorted(boundary_neighbors[index])
                if len(b_neighbors) >= 2:
                    new_positions[index] = 0.75 * vertex + 0.125 * (
                        refined_vertices[b_neighbors[0]] + refined_vertices[b_neighbors[-1]]
                    )
                else:
                    new_positions[index] = vertex
                continue

            n = len(neighbors[index])
            if n == 0:
                new_positions[index] = vertex
                continue
            beta = 3.0 / 16.0 if n == 3 else 3.0 / (8.0 * n)
            neighbor_sum = np.sum(refined_vertices[list(neighbors[index])], axis=0)
            new_positions[index] = (1.0 - n * beta) * vertex + beta * neighbor_sum

        edge_index: dict[tuple[int, int], int] = {}
        new_vertices = [position for position in new_positions]
        for edge, opposites in sorted(edge_opposites.items()):
            i, j = edge
            if len(opposites) == 2:
                edge_point = (
                    3.0 / 8.0 * (refined_vertices[i] + refined_vertices[j])
                    + 1.0 / 8.0 * (refined_vertices[opposites[0]] + refined_vertices[opposites[1]])
                )
            else:
                edge_point = 0.5 * (refined_vertices[i] + refined_vertices[j])
            edge_index[edge] = len(new_vertices)
            new_vertices.append(edge_point)

        new_faces = []
        for i, j, k in refined_faces:
            a = edge_index[tuple(sorted((int(i), int(j))))]
            b = edge_index[tuple(sorted((int(j), int(k))))]
            c = edge_index[tuple(sorted((int(k), int(i))))]
            new_faces.extend(
                [
                    [int(i), a, c],
                    [int(j), b, a],
                    [int(k), c, b],
                    [a, b, c],
                ]
            )
        refined_vertices = np.asarray(new_vertices, dtype=float)
        refined_faces = np.asarray(new_faces, dtype=int)
    return refined_vertices, refined_faces


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return np.nan
    return float(np.interp(quantile * cumulative[-1], cumulative, values))


def cylindrical_asymmetry(
    centroids: np.ndarray,
    weights: np.ndarray,
    radial_bin_count: int = 24,
    cap_radius: float = 30.0,
) -> float:
    r = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    z = centroids[:, 2]
    valid = np.isfinite(r) & np.isfinite(z) & np.isfinite(weights) & (weights > 0) & (r <= cap_radius)
    if np.count_nonzero(valid) < radial_bin_count:
        return np.nan

    r = r[valid]
    z = z[valid]
    weights = weights[valid]
    z_scale = weighted_quantile(z, weights, 0.95) - weighted_quantile(z, weights, 0.05)
    if not np.isfinite(z_scale) or z_scale <= 1e-12:
        return np.nan

    edges = np.linspace(0.0, cap_radius, radial_bin_count + 1)
    residual_sum = 0.0
    weight_sum = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (r >= lo) & (r < hi if hi < cap_radius else r <= hi)
        if np.count_nonzero(in_bin) < 3:
            continue
        bin_weights = weights[in_bin]
        mean_z = np.average(z[in_bin], weights=bin_weights)
        residual_sum += float(np.sum(bin_weights * (z[in_bin] - mean_z) ** 2))
        weight_sum += float(np.sum(bin_weights))
    if weight_sum <= 0:
        return np.nan
    return float(np.sqrt(residual_sum / weight_sum) / z_scale)


def surface_samples_from_control_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    subdivision_iterations: int,
    cap_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered, _ = center_vertices(vertices, np.empty((0, 3), dtype=float))
    half_x = min(abs(float(np.nanmin(centered[:, 0]))), abs(float(np.nanmax(centered[:, 0]))))
    half_y = min(abs(float(np.nanmin(centered[:, 1]))), abs(float(np.nanmax(centered[:, 1]))))
    complete_annulus_radius = min(half_x, half_y) - 2.0
    usable_radius = min(cap_radius, complete_annulus_radius)

    refined_vertices, refined_faces = loop_subdivide(centered, faces, iterations=subdivision_iterations)
    centroids = refined_vertices[refined_faces].mean(axis=1)
    areas = triangle_areas(refined_vertices, refined_faces)
    r = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    in_cap = r <= usable_radius
    return centroids[in_cap], areas[in_cap]


def candidate_rows() -> pd.DataFrame:
    rows = []
    a30b30 = pd.read_csv(A30B30_RUNS).copy()
    a30b30["RigidityNorm"] = a30b30["Rigidity"].map(normalize_rigidity)
    a30b30 = a30b30[a30b30["RigidityNorm"].notna()].copy()
    for _, row in a30b30.iterrows():
        folder = A30B30_BASE / str(row["FolderName"])
        fallback = A30B30_BASE / str(row["FinalVertexFile"])
        rows.append(
            {
                "Ecc": 0.0,
                "Rigidity": float(row["RigidityNorm"]),
                "RunID": int(row["RunNum"]),
                "RelaxArea": float(row["RelaxArea"]),
                "Folder": str(folder),
                "Min_Etot": float(row["Min_Etot"]),
                "Min_step": float(row["Min_step"]),
                "VertexFile": str(nearest_vertex_file(folder, row["Min_step"], fallback=fallback)),
                "FaceFile": str(folder / "face.csv"),
            }
        )

    a30b15 = pd.read_csv(A30B15_RUNS).copy()
    a30b15 = a30b15[
        a30b15["Dataset"].eq("a30b15_restart_partial")
        | (a30b15["Dataset"].eq("regular_geometry") & a30b15["Group"].eq("mesh_a30_b15"))
    ].copy()
    continuation = load_a30b15_continuation_runs()
    if not continuation.empty:
        a30b15 = pd.concat([a30b15, continuation], ignore_index=True, sort=False)
    a30b15["RigidityNorm"] = a30b15["Rigidity"].map(normalize_rigidity)
    a30b15 = a30b15[a30b15["RigidityNorm"].notna()].copy()
    for rigidity in ALLOWED_RIGIDITIES:
        subset = a30b15[np.isclose(a30b15["RigidityNorm"], rigidity)].copy()
        preferred = subset[subset["Dataset"].eq(A30B15_PREFERRED_CONTINUATION_DATASET)]
        if not preferred.empty:
            subset = preferred
        for _, row in subset.iterrows():
            folder = A30B15_BASE / str(row["Folder"])
            fallback = A30B15_BASE / str(row["FinalVertexFile"])
            rows.append(
                {
                    "Ecc": 0.866,
                    "Rigidity": float(rigidity),
                    "RunID": int(row["RunID"]),
                    "RelaxArea": float(row["RelaxArea"]),
                    "Folder": str(folder),
                    "Min_Etot": float(row["Min_Etot"]),
                    "Min_step": float(row["Min_step"]),
                    "VertexFile": str(nearest_vertex_file(folder, row["Min_step"], fallback=fallback)),
                    "FaceFile": str(folder / "face.csv"),
                }
            )
    return pd.DataFrame(rows).sort_values(["Ecc", "Rigidity", "RunID"]).reset_index(drop=True)


def metric_for_row(
    row: pd.Series,
    subdivision_iterations: int,
    cap_radius: float,
    radial_bin_count: int,
) -> tuple[float, int]:
    vertices = read_xyz_csv(Path(row["VertexFile"]))
    faces = read_faces(Path(row["FaceFile"]))
    centroids, weights = surface_samples_from_control_mesh(vertices, faces, subdivision_iterations, cap_radius)
    metric = cylindrical_asymmetry(centroids, weights, radial_bin_count=radial_bin_count, cap_radius=cap_radius)
    return metric, len(centroids)


def bootstrap_joint_minimum(
    candidates: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for (ecc, rigidity), subset in candidates.groupby(["Ecc", "Rigidity"], dropna=False):
        subset = subset[np.isfinite(subset["CylindricalAsymmetry"])].reset_index(drop=True)
        if subset.empty:
            continue
        observed = subset.loc[subset["Min_Etot"].idxmin()].copy()
        values = subset[["Min_Etot", "CylindricalAsymmetry"]].to_numpy(float)
        rng = np.random.default_rng(seed + int(round(1000 * float(ecc))) + int(round(float(rigidity))))
        sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        resampled = values[sample_indices]
        min_indices = np.argmin(resampled[:, :, 0], axis=1)
        selected_metrics = resampled[np.arange(n_bootstrap), min_indices, 1]
        rows.append(
            {
                "Ecc": float(ecc),
                "Rigidity": float(rigidity),
                "CylindricalAsymmetry": float(observed["CylindricalAsymmetry"]),
                "StdErrCylindricalAsymmetry": float(np.std(selected_metrics, ddof=1)),
                "BootstrapMetricP025": float(np.percentile(selected_metrics, 2.5)),
                "BootstrapMetricP975": float(np.percentile(selected_metrics, 97.5)),
                "Runs": int(len(subset)),
                "SurfaceFaceSamples": int(observed["SurfaceFaceSamples"]),
                "SubdivisionIterations": int(observed["SubdivisionIterations"]),
                "CapRadius": float(observed["CapRadius"]),
                "RadialBins": int(observed["RadialBins"]),
                "RunID": int(observed["RunID"]),
                "RelaxArea": float(observed["RelaxArea"]),
                "VertexFile": observed["VertexFile"],
                "FaceFile": observed["FaceFile"],
                "Min_Etot": float(observed["Min_Etot"]),
                "Min_step": float(observed["Min_step"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["Ecc", "Rigidity"]).reset_index(drop=True)


def compute_summary(
    subdivision_iterations: int = 3,
    cap_radius: float = 30.0,
    radial_bin_count: int = 24,
    n_bootstrap: int = 10000,
) -> pd.DataFrame:
    cache_path = RESULTS / "cylindrical_symmetry_candidate_trajectories.csv"
    candidates = candidate_rows()
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame()
    metric_rows = []
    for index, row in candidates.iterrows():
        cached_match = pd.DataFrame()
        if not cached.empty:
            cached_match = cached[
                cached["VertexFile"].eq(row["VertexFile"])
                & cached["FaceFile"].eq(row["FaceFile"])
                & np.isclose(cached["SubdivisionIterations"].astype(float), subdivision_iterations)
                & np.isclose(cached["CapRadius"].astype(float), cap_radius)
                & np.isclose(cached["RadialBins"].astype(float), radial_bin_count)
            ]
        if not cached_match.empty:
            out = cached_match.iloc[0].to_dict()
            print(
                f"[{index + 1}/{len(candidates)}] cached e={row['Ecc']:.3g} "
                f"K={row['Rigidity']:g} run={int(row['RunID'])}"
            )
        else:
            print(
                f"[{index + 1}/{len(candidates)}] computing e={row['Ecc']:.3g} "
                f"K={row['Rigidity']:g} run={int(row['RunID'])}",
                flush=True,
            )
            metric, n_samples = metric_for_row(row, subdivision_iterations, cap_radius, radial_bin_count)
            out = row.to_dict()
            out.update(
                {
                    "CylindricalAsymmetry": metric,
                    "SurfaceFaceSamples": n_samples,
                    "SubdivisionIterations": subdivision_iterations,
                    "CapRadius": cap_radius,
                    "RadialBins": radial_bin_count,
                }
            )
        metric_rows.append(out)
        pd.DataFrame(metric_rows).to_csv(cache_path, index=False)
    candidates_with_metrics = pd.DataFrame(metric_rows)
    candidates_with_metrics.to_csv(cache_path, index=False)
    summary = bootstrap_joint_minimum(candidates_with_metrics, n_bootstrap=n_bootstrap, seed=20260616)
    summary.to_csv(RESULTS / "cylindrical_symmetry_selected_minima.csv", index=False)
    summary.to_csv(RESULTS / "cylindrical_symmetry_surface_summary.csv", index=False)
    return summary


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
        out = PLOTS / f"cylindrical_symmetry_vs_rigidity.{suffix}"
        fig.savefig(out, dpi=450 if suffix == "png" else None, bbox_inches="tight")
        print(out)


def main() -> None:
    summary = compute_summary()
    plot_summary(summary)


if __name__ == "__main__":
    main()
