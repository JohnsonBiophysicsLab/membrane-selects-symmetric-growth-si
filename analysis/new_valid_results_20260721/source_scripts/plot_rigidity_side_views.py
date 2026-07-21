from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection


ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
RESULTS = ROOT / "results"
PLOTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

A30B30_RUNS = Path("/Users/yueying/Downloads/change_stiffness/results/rigidity_results.csv")
A30B30_BASE = Path("/Users/yueying/Downloads/change_stiffness")
A30B15_RUNS = RESULTS / "all_runs.csv"
A30B15_BASE = ROOT
A30B15_CONTINUATION_BASE = (
    ROOT
    / "fetched"
    / "idealized_lattice"
    / "change_stiffness_a30b15_restart_20260528"
)
A30B15_PREFERRED_CONTINUATION_DATASET = "a30b15_s1000_from_s316_anneal_20260616"
A30B15_CONTINUATION_RUNS = [
    (
        A30B15_PREFERRED_CONTINUATION_DATASET,
        "s1000_from_s316_anneal_20260616",
        A30B15_CONTINUATION_BASE / "s1000_from_s316_anneal_20260616",
    ),
    (
        "a30b15_s1000_continue_20260609",
        "s1000_continue_20260609",
        A30B15_CONTINUATION_BASE / "s1000_continue_20260609",
    ),
]

ALLOWED_RIGIDITIES = [10.0, 31.6, 100.0, 316.0, 1000.0]
VERTEX_RE = re.compile(r"^vertex(?P<frame>\d+)\.csv$")
GAG_RE = re.compile(r"^gag_scaffold_(?P<step>\d+)\.dat$")
ITER_RE = re.compile(
    r"Step:\s*(?P<step>\d+).*?"
    r"energy=\s*(?P<Etot>[-+0-9.eE]+)\.?\s*"
    r"eHBond=\s*(?P<eHBond>[-+0-9.eE]+)\.?\s*"
    r"eGag=\s*(?P<eGag>[-+0-9.eE]+)\.?\s*"
    r"eIdeal=\s*(?P<eIdeal>[-+0-9.eE]+)\.?\s*"
    r"meanF=\s*(?P<meanF>[-+0-9.eE]+)\.?\s*"
    r"area=\s*(?P<Area>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([^#]+)")


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "savefig.transparent": False,
    }
)


def normalize_rigidity(value: float) -> float | None:
    for rigidity in ALLOWED_RIGIDITIES:
        if np.isclose(float(value), rigidity, rtol=0.0, atol=0.7):
            return rigidity
    return None


def parse_float(text: str) -> float:
    return float(str(text).strip().rstrip("."))


def parse_params(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    if not path.exists():
        return params
    for line in path.read_text(errors="ignore").splitlines():
        match = PARAM_RE.match(line)
        if match:
            params[match.group(1)] = match.group(2).strip()
    return params


def param_float(params: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(params[key])
    except (KeyError, ValueError):
        return default


def parse_energy_logs(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="ignore").splitlines():
            match = ITER_RE.search(line)
            if not match:
                continue
            row = {key: parse_float(value) for key, value in match.groupdict().items()}
            row["step"] = int(row["step"])
            rows.append(row)
    return pd.DataFrame(rows)


def trial_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def load_a30b15_continuation_runs() -> pd.DataFrame:
    """Parse fetched continuation logs, if present, as preferred K=1000 rows."""
    rows = []
    for dataset_name, group_name, root in A30B15_CONTINUATION_RUNS:
        if not root.exists():
            continue
        for trial_dir in sorted(root.glob("trial_*"), key=trial_number):
            logs = sorted(trial_dir.glob("output*.log"))
            if not logs:
                continue
            data = parse_energy_logs(logs)
            if data.empty:
                continue
            params = parse_params(trial_dir / "input.params")
            min_row = data.loc[data["Etot"].idxmin()]
            final_row = data.iloc[-1]
            run_id = trial_number(trial_dir)
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Group": group_name,
                    "RunID": run_id,
                    "Folder": str(trial_dir.relative_to(ROOT)),
                    "MajorAxis": 30.0,
                    "MinorAxis": 15.0,
                    "Ecc": 0.8660254037844386,
                    "Ngags": 76.0,
                    "Rigidity": 1000.0,
                    "RigidityLabel": "s1000",
                    "RelaxArea": param_float(params, "relaxArea"),
                    "FinalVertexFile": str(nearest_vertex_file(trial_dir, final_row["step"]).relative_to(ROOT)),
                    "Final_step": float(final_row["step"]),
                    "Final_Etot": float(final_row["Etot"]),
                    "Final_eHBond": float(final_row["eHBond"]),
                    "Final_eGag": float(final_row["eGag"]),
                    "Final_eIdeal": float(final_row["eIdeal"]),
                    "Final_meanF": float(final_row["meanF"]),
                    "Final_Area": float(final_row["Area"]),
                    "Min_step": float(min_row["step"]),
                    "Min_Etot": float(min_row["Etot"]),
                    "Min_eHBond": float(min_row["eHBond"]),
                    "Min_eGag": float(min_row["eGag"]),
                    "Min_eIdeal": float(min_row["eIdeal"]),
                    "Min_meanF": float(min_row["meanF"]),
                    "Min_Area": float(min_row["Area"]),
                }
            )
    return pd.DataFrame(rows)


def read_xyz_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if {"x", "y", "z"}.issubset(df.columns):
        return df[["x", "y", "z"]].to_numpy(float)
    return pd.read_csv(path, header=None, names=["x", "y", "z"]).to_numpy(float)


def read_faces(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if {"i", "j", "k"}.issubset(df.columns):
        faces = df[["i", "j", "k"]].to_numpy(int)
    else:
        faces = pd.read_csv(path, header=None, names=["i", "j", "k"]).to_numpy(int)
    return faces


def vertex_step(path: Path) -> int | None:
    if path.name == "vertexfinal.csv":
        return None
    match = VERTEX_RE.match(path.name)
    if not match:
        return None
    return int(match.group("frame")) * 100


def nearest_vertex_file(folder: Path, target_step: float, fallback: Path | None = None) -> Path:
    candidates: list[tuple[float, Path]] = []
    for path in folder.glob("vertex*.csv"):
        step = vertex_step(path)
        if step is not None:
            candidates.append((abs(step - float(target_step)), path))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    if fallback is not None and fallback.exists():
        return fallback
    raise FileNotFoundError(f"no vertex snapshots in {folder}")


def nearest_gag_file(folder: Path, target_step: float) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for path in folder.glob("gag_scaffold_*.dat"):
        match = GAG_RE.match(path.name)
        if match:
            step = int(match.group("step"))
            candidates.append((abs(step - float(target_step)), path))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def read_gag_points(path: Path | None, fallback_csv: Path | None = None) -> np.ndarray:
    if path is not None and path.exists():
        data = json.loads(path.read_text())
        molecules = data.get("molecules", {})
        points = []
        for molecule in molecules.values():
            if "com" in molecule:
                points.append(molecule["com"])
        if points:
            return np.asarray(points, dtype=float)
    if fallback_csv is not None and fallback_csv.exists():
        return read_xyz_csv(fallback_csv)
    return np.empty((0, 3), dtype=float)


def select_a30b30() -> pd.DataFrame:
    data = pd.read_csv(A30B30_RUNS).copy()
    data["RigidityNorm"] = data["Rigidity"].map(normalize_rigidity)
    data = data[data["RigidityNorm"].notna()].copy()
    rows = []
    for rigidity in ALLOWED_RIGIDITIES:
        subset = data[np.isclose(data["RigidityNorm"], rigidity)]
        if subset.empty:
            continue
        row = subset.loc[subset["Min_Etot"].idxmin()].copy()
        folder = A30B30_BASE / str(row["FolderName"])
        fallback = A30B30_BASE / str(row["FinalVertexFile"])
        vertex = nearest_vertex_file(folder, row["Min_step"], fallback=fallback)
        gag = nearest_gag_file(folder, row["Min_step"])
        rows.append(
            {
                "Ecc": 0.0,
                "Rigidity": rigidity,
                "Folder": str(folder),
                "Min_Etot": float(row["Min_Etot"]),
                "Min_step": float(row["Min_step"]),
                "VertexFile": str(vertex),
                "FaceFile": str(folder / "face.csv"),
                "GagFile": str(gag) if gag else "",
                "GagFallbackCsv": str(folder / "mc_mesh_a30_b30.csv"),
            }
        )
    return pd.DataFrame(rows)


def select_a30b15() -> pd.DataFrame:
    data = pd.read_csv(A30B15_RUNS).copy()
    data = data[
        data["Dataset"].eq("a30b15_restart_partial")
        | (data["Dataset"].eq("regular_geometry") & data["Group"].eq("mesh_a30_b15"))
    ].copy()
    continuation = load_a30b15_continuation_runs()
    if not continuation.empty:
        data = pd.concat([data, continuation], ignore_index=True, sort=False)
    data["RigidityNorm"] = data["Rigidity"].map(normalize_rigidity)
    data = data[data["RigidityNorm"].notna()].copy()
    rows = []
    for rigidity in ALLOWED_RIGIDITIES:
        subset = data[np.isclose(data["RigidityNorm"], rigidity)]
        if subset.empty:
            continue
        preferred = subset[subset["Dataset"].eq(A30B15_PREFERRED_CONTINUATION_DATASET)]
        if not preferred.empty:
            subset = preferred
        row = subset.loc[subset["Min_Etot"].idxmin()].copy()
        folder = A30B15_BASE / str(row["Folder"])
        fallback = A30B15_BASE / str(row["FinalVertexFile"])
        vertex = nearest_vertex_file(folder, row["Min_step"], fallback=fallback)
        gag = nearest_gag_file(folder, row["Min_step"])
        rows.append(
            {
                "Ecc": 0.866,
                "Rigidity": rigidity,
                "Folder": str(folder),
                "Min_Etot": float(row["Min_Etot"]),
                "Min_step": float(row["Min_step"]),
                "VertexFile": str(vertex),
                "FaceFile": str(folder / "face.csv"),
                "GagFile": str(gag) if gag else "",
                "GagFallbackCsv": str(folder / "mc_mesh_a30_b15.csv"),
            }
        )
    return pd.DataFrame(rows)


def center_vertices(vertices: np.ndarray, gag_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = vertices.copy()
    center_xy = np.nanmean(centered[:, :2], axis=0)
    centered[:, 0] -= center_xy[0]
    centered[:, 1] -= center_xy[1]
    centered[:, 2] -= np.nanmin(centered[:, 2])
    gags = gag_points.copy()
    if len(gags):
        gags[:, 0] -= center_xy[0]
        gags[:, 1] -= center_xy[1]
        gags[:, 2] -= np.nanmin(vertices[:, 2])
    return centered, gags


def projected_polygons(vertices: np.ndarray, faces: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    # Side view looking along the x-axis: horizontal = y, vertical = z.
    y = vertices[:, 1]
    z = vertices[:, 2]
    face_depth = vertices[faces, 0].mean(axis=1)
    order = np.argsort(face_depth)
    polygons = [np.column_stack([y[tri], z[tri]]) for tri in faces[order]]
    colors = vertices[faces[order], 2].mean(axis=1)
    return polygons, colors, order


def draw_panel(ax: plt.Axes, row: pd.Series, norm: mpl.colors.Normalize) -> None:
    vertices = read_xyz_csv(Path(row["VertexFile"]))
    faces = read_faces(Path(row["FaceFile"]))
    gag_file = Path(row["GagFile"]) if row["GagFile"] else None
    fallback = Path(row["GagFallbackCsv"]) if row["GagFallbackCsv"] else None
    gags = read_gag_points(gag_file, fallback_csv=fallback)
    vertices, gags = center_vertices(vertices, gags)
    polygons, colors, _ = projected_polygons(vertices, faces)
    collection = PolyCollection(
        polygons,
        array=colors,
        cmap="viridis",
        norm=norm,
        linewidths=0.0,
        edgecolors="none",
        alpha=0.96,
        rasterized=True,
    )
    ax.add_collection(collection)
    if len(gags):
        ax.scatter(
            gags[:, 1],
            gags[:, 2],
            s=4.0,
            c="#16745c",
            alpha=0.34,
            linewidths=0,
            zorder=5,
            rasterized=True,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def main() -> None:
    selected = pd.concat([select_a30b30(), select_a30b15()], ignore_index=True)
    selected = selected.sort_values(["Ecc", "Rigidity"]).reset_index(drop=True)
    selected.to_csv(RESULTS / "rigidity_side_view_selected_minima.csv", index=False)

    loaded = []
    distance_rows = []
    for _, row in selected.iterrows():
        vertices = read_xyz_csv(Path(row["VertexFile"]))
        gags = read_gag_points(
            Path(row["GagFile"]) if row["GagFile"] else None,
            fallback_csv=Path(row["GagFallbackCsv"]) if row["GagFallbackCsv"] else None,
        )
        vertices, gags = center_vertices(vertices, gags)
        loaded.append((row, vertices, gags))
        if len(gags):
            vertex_xy = vertices[:, :2]
            distances = []
            dz_values = []
            for point in gags:
                nearest = int(np.argmin(np.sum((vertex_xy - point[:2]) ** 2, axis=1)))
                distances.append(float(np.linalg.norm(vertices[nearest] - point)))
                dz_values.append(float(point[2] - vertices[nearest, 2]))
            distance_rows.append(
                {
                    "Ecc": row["Ecc"],
                    "Rigidity": row["Rigidity"],
                    "MedianNearestDistance": float(np.median(distances)),
                    "MinNearestDistance": float(np.min(distances)),
                    "MaxNearestDistance": float(np.max(distances)),
                    "MedianDzScaffoldMinusMembrane": float(np.median(dz_values)),
                    "MinDzScaffoldMinusMembrane": float(np.min(dz_values)),
                    "MaxDzScaffoldMinusMembrane": float(np.max(dz_values)),
                    "ExpectedLbondNm": 9.0,
                }
            )
    pd.DataFrame(distance_rows).to_csv(RESULTS / "rigidity_side_view_scaffold_distance_check.csv", index=False)

    raised_y_extents = []
    z_min = min(float(vertices[:, 2].min()) for _, vertices, _ in loaded)
    z_max = max(float(vertices[:, 2].max()) for _, vertices, _ in loaded)
    for _, vertices, gags in loaded:
        raised = vertices[:, 2] > 0.05 * float(vertices[:, 2].max())
        if np.any(raised):
            raised_y_extents.extend([float(vertices[raised, 1].min()), float(vertices[raised, 1].max())])
        if len(gags):
            raised_y_extents.extend([float(gags[:, 1].min()), float(gags[:, 1].max())])
            z_min = min(z_min, float(gags[:, 2].min()))
            z_max = max(z_max, float(gags[:, 2].max()))
    y_abs = max(abs(value) for value in raised_y_extents) + 4.0
    y_limits = (-y_abs, y_abs)
    z_limits = (z_min - 0.8, z_max + 1.0)
    norm = mpl.colors.Normalize(vmin=z_limits[0], vmax=z_limits[1])

    fig, axes = plt.subplots(2, 5, figsize=(14.2, 2.15), constrained_layout=False)
    fig.patch.set_facecolor("white")

    for row_index, ecc in enumerate([0.0, 0.866]):
        for col_index, rigidity in enumerate(ALLOWED_RIGIDITIES):
            ax = axes[row_index, col_index]
            match = selected[
                np.isclose(selected["Ecc"], ecc) & np.isclose(selected["Rigidity"], rigidity)
            ]
            if match.empty:
                ax.axis("off")
                continue
            draw_panel(ax, match.iloc[0], norm)
            ax.set_xlim(*y_limits)
            ax.set_ylim(*z_limits)
            if row_index == 0:
                ax.set_title(f"K = {rigidity:g}", fontsize=12, pad=2)
            if col_index == 0:
                ax.text(
                    0.0,
                    0.5,
                    f"e = {ecc:g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=12,
                )

    fig.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.03, wspace=0.03, hspace=0.02)
    for suffix in ("png", "svg"):
        out = PLOTS / f"rigidity_side_views_equal_scale.{suffix}"
        fig.savefig(out, dpi=450 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.02)
        print(out)


if __name__ == "__main__":
    main()
