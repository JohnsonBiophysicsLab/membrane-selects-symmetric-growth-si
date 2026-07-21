from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from plot_curvature_difference_rigidity import (
    ALLOWED_RIGIDITIES,
    PLOTS,
    RESULTS,
    cap_mask,
    center_vertices,
    curvature_difference,
    face_values,
    read_faces,
    read_xyz_csv,
    topdown_polygons,
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


def load_selected_maps() -> list[dict]:
    selected_path = RESULTS / "curvature_difference_selected_minima.csv"
    selected = pd.read_csv(selected_path).sort_values(["Ecc", "Rigidity"]).reset_index(drop=True)
    maps = []
    for _, row in selected.iterrows():
        vertices = read_xyz_csv(Path(row["VertexFile"]))
        faces = read_faces(Path(row["FaceFile"]))
        vertices, _ = center_vertices(vertices, np.empty((0, 3), dtype=float))
        mean_curv, _gaussian_curv, _kappa_diff = curvature_difference(vertices, faces)
        cap = cap_mask(vertices, radius=30.0)
        maps.append(
            {
                "row": row,
                "vertices": vertices,
                "faces": faces,
                "mean_curv": mean_curv,
                "mask": cap & np.isfinite(mean_curv),
            }
        )
    return maps


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
            collection = PolyCollection(
                topdown_polygons(vertices, faces_to_plot),
                array=face_values(values, faces_to_plot),
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


def main() -> None:
    maps = load_selected_maps()
    plot_mean_curvature_heatmaps(maps)


if __name__ == "__main__":
    main()
