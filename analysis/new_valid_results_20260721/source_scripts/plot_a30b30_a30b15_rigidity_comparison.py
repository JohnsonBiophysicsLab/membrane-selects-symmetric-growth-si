import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_rigidity_side_views import (
    A30B15_PREFERRED_CONTINUATION_DATASET,
    load_a30b15_continuation_runs,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

A30B30_RUNS = Path("/Users/yueying/Downloads/change_stiffness/results/rigidity_results.csv")
A30B30_RECOMPUTED = ROOT / "results" / "a30b30_min_etot_old_cap_grouped_minima.csv"
A30B15_RUNS = ROOT / "results" / "all_runs.csv"
A30B15_RECOMPUTED = ROOT / "results" / "a30b15_min_etot_old_cap_grouped_minima.csv"
A30B15_RUNS_WITH_CURVATURE = ROOT / "results" / "a30b15_min_etot_old_cap_runs_with_curvature.csv"
ALLOWED_RIGIDITIES = np.array([10.0, 31.6, 100.0, 316.0, 1000.0])

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 9,
        "axes.linewidth": 1.2,
    }
)


def old_interval_errors(df: pd.DataFrame, value: str, low: str, high: str) -> np.ndarray:
    """Old ~/Downloads/change_stiffness CSV stores some bootstrap intervals as bounds."""
    center = df[value].to_numpy(float)
    lo = df[low].to_numpy(float)
    hi = df[high].to_numpy(float)
    err_l = np.maximum(0.0, center - lo)
    err_u = np.maximum(0.0, hi - center)
    return np.vstack([err_l, err_u])


def keep_allowed_rigidities(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Keep only the intended stiffness grid; map legacy s31 label onto 31.6."""
    df = df.copy()
    legacy_s31 = np.isclose(df["Rigidity"].astype(float), 31.0)
    df.loc[legacy_s31, "Rigidity"] = 31.6

    allowed = np.zeros(len(df), dtype=bool)
    for rigidity in ALLOWED_RIGIDITIES:
        allowed |= np.isclose(df["Rigidity"].astype(float), rigidity)
    dropped = df.loc[~allowed, "Rigidity"].tolist()
    if dropped:
        print(f"Dropped off-grid {name} rigidities: {dropped}")
    return df.loc[allowed].copy()


def read_vertices_array(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None, names=["x", "y", "z"]).to_numpy(float)


def read_faces_array(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None, names=["i", "j", "k"]).to_numpy(int)


def vertex_mean_curvature(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Match the old change_stiffness notebook cotangent-Laplacian estimate."""
    n = len(vertices)
    lap = np.zeros((n, 3), dtype=float)
    area = np.zeros(n, dtype=float)

    def cotangent(u: np.ndarray, v: np.ndarray) -> float:
        cross_norm = np.linalg.norm(np.cross(u, v))
        if cross_norm < 1e-12:
            return 0.0
        return float(np.dot(u, v) / cross_norm)

    for tri in faces:
        i, j, k = tri
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        face_area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
        if face_area <= 1e-12:
            continue
        area[[i, j, k]] += face_area / 3.0

        cot_i = cotangent(vj - vi, vk - vi)
        cot_j = cotangent(vi - vj, vk - vj)
        cot_k = cotangent(vi - vk, vj - vk)

        lap[i] += cot_k * (vj - vi) + cot_j * (vk - vi)
        lap[j] += cot_k * (vi - vj) + cot_i * (vk - vj)
        lap[k] += cot_j * (vi - vk) + cot_i * (vj - vk)

    valid = area > 1e-12
    delta = np.zeros_like(lap)
    delta[valid] = lap[valid] / (2.0 * area[valid, None])
    return 0.5 * np.linalg.norm(delta, axis=1)


def cap_curvature_summary(vertex_path: Path, face_path: Path, cap_radius: float = 30.0) -> float:
    """Match the old a30_b30 cap definition: uncentered projected radius <= 30 nm."""
    if not vertex_path.exists() or not face_path.exists():
        return np.nan
    vertices = read_vertices_array(vertex_path)
    faces = read_faces_array(face_path)
    curv = vertex_mean_curvature(vertices, faces)
    radial = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
    cap_mask = radial <= cap_radius
    if not np.any(cap_mask):
        return np.nan
    return float(np.nanmean(curv[cap_mask]))


def bootstrap_by_min_etot(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    value_cols = ["MeanCurvatureCap"]
    for keys, subset in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        observed = subset.loc[subset["Min_Etot"].idxmin()].copy()
        arr = subset[["Min_Etot", *value_cols]].to_numpy(float)
        seed_key = int(float(observed["Rigidity"])) + sum(len(str(key)) for key in keys)
        rng = np.random.default_rng(20260522 + seed_key)
        sample_indices = rng.integers(0, len(arr), size=(10000, len(arr)))
        resampled = arr[sample_indices]
        min_indices = np.argmin(resampled[:, :, 0], axis=1)
        min_rows = resampled[np.arange(len(resampled)), min_indices]
        metric_values = min_rows[:, 1]
        q1, q3 = np.nanpercentile(metric_values, [25, 75])
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            keep = (metric_values >= q1 - 1.5 * iqr) & (metric_values <= q3 + 1.5 * iqr)
            metric_values_no_outliers = metric_values[keep]
        else:
            keep = np.ones(len(metric_values), dtype=bool)
            metric_values_no_outliers = metric_values
        if len(metric_values_no_outliers) == 0:
            metric_values_no_outliers = metric_values

        out = observed.to_dict()
        out["Runs"] = len(subset)
        out["Min_Etot_err_u"] = np.percentile(min_rows[:, 0], 95) - float(observed["Min_Etot"])
        out["Min_Etot_se"] = float(np.nanstd(min_rows[:, 0], ddof=1))
        out["MeanCurvatureCap_se_raw"] = float(np.nanstd(metric_values, ddof=1))
        out["MeanCurvatureCap_se"] = float(np.nanstd(metric_values_no_outliers, ddof=1))
        out["MeanCurvatureCap_bootstrap_outliers_removed"] = int(np.count_nonzero(~keep))
        alpha = (1 - 0.90) / 2
        out["MeanCurvatureCap_err_l"] = np.percentile(metric_values_no_outliers, 100 * alpha)
        out["MeanCurvatureCap_err_u"] = np.percentile(metric_values_no_outliers, 100 * (1 - alpha))
        rows.append(out)
    return pd.DataFrame(rows)


def recompute_a30b30_min_etot_old_cap() -> pd.DataFrame:
    force = os.environ.get("SLIMED_FORCE_RECOMPUTE_A30B30", "").lower() in {"1", "true", "yes"}
    if A30B30_RECOMPUTED.exists() and not force:
        if A30B30_RECOMPUTED.stat().st_mtime >= A30B30_RUNS.stat().st_mtime:
            return keep_allowed_rigidities(pd.read_csv(A30B30_RECOMPUTED), name="a30_b30 cached").sort_values("Rigidity")

    data = keep_allowed_rigidities(pd.read_csv(A30B30_RUNS), name="a30_b30 raw")
    out = bootstrap_by_min_etot(data, ["RunSet", "Rigidity"])
    out = keep_allowed_rigidities(out, name="a30_b30 recomputed").sort_values("Rigidity")
    out.to_csv(A30B30_RECOMPUTED, index=False)
    return out


def recompute_a30b15_min_etot_old_cap() -> pd.DataFrame:
    force = os.environ.get("SLIMED_FORCE_RECOMPUTE_A30B15", "").lower() in {"1", "true", "yes"}
    if A30B15_RECOMPUTED.exists() and not force:
        if A30B15_RECOMPUTED.stat().st_mtime >= A30B15_RUNS.stat().st_mtime:
            return keep_allowed_rigidities(pd.read_csv(A30B15_RECOMPUTED), name="a30_b15 cached").sort_values("Rigidity")

    runs = pd.read_csv(A30B15_RUNS)
    restart = runs[runs["Dataset"].eq("a30b15_restart_partial")].copy()
    default = runs[runs["Dataset"].eq("regular_geometry") & runs["Group"].eq("mesh_a30_b15")].copy()
    data = keep_allowed_rigidities(pd.concat([restart, default], ignore_index=True), name="a30_b15 raw")
    continuation = load_a30b15_continuation_runs()
    if not continuation.empty:
        data = data[~np.isclose(data["Rigidity"].astype(float), 1000.0)].copy()
        preferred = continuation[continuation["Dataset"].eq(A30B15_PREFERRED_CONTINUATION_DATASET)].copy()
        if not preferred.empty:
            continuation = preferred
        data = keep_allowed_rigidities(pd.concat([data, continuation], ignore_index=True, sort=False), name="a30_b15 with continuation")
    data = data[data["FinalVertexFile"].notna() & data["Folder"].notna()].copy()

    curvatures = []
    min_vertex_files = []
    for _, row in data.iterrows():
        folder = ROOT / str(row["Folder"])
        fallback = ROOT / str(row["FinalVertexFile"])
        from plot_rigidity_side_views import nearest_vertex_file

        vertex_path = nearest_vertex_file(folder, row["Min_step"], fallback=fallback)
        min_vertex_files.append(str(vertex_path.relative_to(ROOT)))
        curvatures.append(cap_curvature_summary(vertex_path, folder / "face.csv", cap_radius=30.0))
    data["MeanCurvatureCap"] = curvatures
    data["MinVertexFileForCurvature"] = min_vertex_files
    data.to_csv(A30B15_RUNS_WITH_CURVATURE, index=False)

    out = bootstrap_by_min_etot(data, ["Group", "Rigidity"])
    out = keep_allowed_rigidities(out, name="a30_b15 recomputed").sort_values("Rigidity")
    out.to_csv(A30B15_RECOMPUTED, index=False)
    return out


def main() -> None:
    a30b30 = recompute_a30b30_min_etot_old_cap()
    a30b15 = recompute_a30b15_min_etot_old_cap()
    a30b15 = a30b15[a30b15["Runs"] >= 16].copy()
    a30b30 = a30b30.sort_values("Rigidity")
    a30b15 = a30b15.drop_duplicates("Rigidity", keep="last").sort_values("Rigidity")

    fig, (ax_e, ax_c) = plt.subplots(2, 1, figsize=(5.2, 6.2), sharex=True)
    axes = [ax_e, ax_c]

    ax_e.errorbar(
        a30b30["Rigidity"],
        a30b30["Min_Etot"],
        yerr=[np.zeros(len(a30b30)), a30b30["Min_Etot_err_u"].to_numpy(float)],
        fmt="o",
        color="black",
        ecolor="0.2",
        elinewidth=1.4,
        capsize=5.5,
        capthick=1.6,
        ms=6,
        label="0",
    )
    if not a30b15.empty:
        ax_e.errorbar(
            a30b15["Rigidity"],
            a30b15["Min_Etot"],
            yerr=[np.zeros(len(a30b15)), a30b15["Min_Etot_err_u"].to_numpy(float)],
            fmt="s",
            color="0.45",
            ecolor="0.45",
            elinewidth=1.4,
            capsize=5.5,
            capthick=1.6,
            ms=6,
            label="0.866",
        )

    # Reference lines copied stylistically from the previous rigidity figure.
    ref_box = {"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.6}
    ax_e.axhline(216.5, color="0.25", linestyle="-.", linewidth=1.0)
    ax_e.axhline(214.2, color="0.45", linestyle="--", linewidth=1.0)
    ax_e.text(9.0, 217.0, "simulation min (rigid body)", fontsize=6, va="bottom", color="0.25", bbox=ref_box)
    ax_e.text(9.0, 213.6, "theoretical min (rigid body)", fontsize=6, va="top", color="0.35", bbox=ref_box)
    ax_e.set_ylabel(r"Minimum Total Energy (pN$\cdot$nm)")
    ax_e.set_ylim(88, 224)

    ax_c.errorbar(
        a30b30["Rigidity"],
        a30b30["MeanCurvatureCap"],
        yerr=a30b30["MeanCurvatureCap_se"].to_numpy(float),
        fmt="o",
        color="black",
        ecolor="0.2",
        elinewidth=1.4,
        capsize=5.5,
        capthick=1.6,
        ms=6,
        label="0",
    )
    if not a30b15.empty:
        ax_c.errorbar(
            a30b15["Rigidity"],
            a30b15["MeanCurvatureCap"],
            yerr=a30b15["MeanCurvatureCap_se"].to_numpy(float),
            fmt="s",
            color="0.45",
            ecolor="0.45",
            elinewidth=1.4,
            capsize=5.5,
            capthick=1.6,
            ms=6,
            label="0.866",
        )

    ax_c.set_ylabel(r"Mean Curvature of Cap (nm$^{-1}$)")
    ax_c.set_xlabel(r"Protein Lattice Rigidity (pN$\cdot$nm or pN/nm)")
    ax_c.set_ylim(0.0120, 0.01615)
    ax_e.yaxis.set_label_coords(-0.15, 0.5)
    ax_c.yaxis.set_label_coords(-0.15, 0.5)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(8, 1200)
        ax.tick_params(axis="both", which="major", length=6, width=1.1, direction="out")
        ax.tick_params(axis="both", which="minor", length=3, width=1.0, direction="out")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)

    handles, labels = ax_e.get_legend_handles_labels()
    legend = ax_e.legend(
        handles,
        labels,
        title="Eccentricity",
        frameon=True,
        fancybox=False,
        loc="center right",
        bbox_to_anchor=(0.98, 0.40),
        ncol=1,
        handlelength=1.4,
        borderpad=0.2,
        labelspacing=0.5,
        handletextpad=0.6,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.82)
    legend.get_frame().set_edgecolor("0.35")
    legend.get_frame().set_linewidth(0.8)
    legend.get_title().set_fontsize(9)
    fig.subplots_adjust(hspace=0.12)
    png = OUT / "a30b30_a30b15_rigidity_energy_curvature.png"
    svg = OUT / "a30b30_a30b15_rigidity_energy_curvature.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
