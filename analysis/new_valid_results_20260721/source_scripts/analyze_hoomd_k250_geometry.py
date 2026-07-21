#!/usr/bin/env python3
"""Canonical geometry analysis for HOOMD membrane height-field trajectories.

The k250 trajectories store an 81 x 81 membrane height field in
out_simulation/mem_traj.json.  This script averages the second half of each
trajectory and computes geometry metrics on the resulting canonical mean
surface.  It also records per-frame feature fluctuations over the same second
half window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - handled at runtime on the cluster
    least_squares = None

try:
    import gsd.hoomd as gsd_hoomd
except Exception:  # pragma: no cover - optional; enables Gag-centered cap masks
    gsd_hoomd = None


SPHERE_RADIUS_NM = 50.0
MAX_FIT_SAMPLES = 3500


ELLIPSOID_KEYS = [
    "EllipsoidCenterXNm",
    "EllipsoidCenterYNm",
    "EllipsoidCenterZNm",
    "EllipsoidAxisXNm",
    "EllipsoidAxisYNm",
    "EllipsoidAxisZNm",
    "EllipsoidAxisMinNm",
    "EllipsoidAxisMidNm",
    "EllipsoidAxisMaxNm",
    "EllipsoidAxisMeanNm",
    "EllipsoidAxisSpreadNm",
    "SphereAxisRmsDeviationNm",
    "SphereAxisRmsDeviationFraction",
    "SphereAxisRmsDeviationPercent",
    "SphereAxisMeanAbsDeviationNm",
    "SphereAxisMaxAbsDeviationNm",
    "EllipsoidFitRmseNm",
    "EllipsoidFitMedianAbsResidualNm",
    "EllipsoidFitConverged",
    "EllipsoidFitCost",
    "EllipsoidApexRadiusXNm",
    "EllipsoidApexRadiusYNm",
    "EllipsoidApexRadiusMeanNm",
    "EllipsoidApexRadiusSpreadNm",
    "EllipsoidApexRadiusAnisotropyFraction",
    "SphereCurvatureRadiusRmsDeviationNm",
    "SphereCurvatureRadiusRmsDeviationFraction",
    "SphereCurvatureRadiusRmsDeviationPercent",
    "ClosestSphereRadiusNm",
    "ClosestSphereRadiusRmsDeviationNm",
    "ClosestSphereRadiusRmsDeviationFraction",
    "ClosestSphereRadiusRmsDeviationPercent",
    "EllipsoidApexEccentricity",
]


QUADRATIC_KEYS = [
    "QuadraticFitRmseNm",
    "QuadraticPrincipalCurvature1PerNm",
    "QuadraticPrincipalCurvature2PerNm",
    "QuadraticApexRadius1Nm",
    "QuadraticApexRadius2Nm",
    "QuadraticClosestSphereRadiusNm",
    "QuadraticClosestSphereRadiusRmsDeviationNm",
    "QuadraticClosestSphereRadiusRmsDeviationPercent",
    "QuadraticSphere50RadiusRmsDeviationNm",
    "QuadraticSphere50RadiusRmsDeviationPercent",
    "QuadraticApexEccentricity",
]


def finite_or_nan(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def frame_to_grid(frame: Any, n: int) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float64)
    if arr.shape == (n, n):
        return arr
    if arr.size == n * n:
        return arr.reshape((n, n))
    raise ValueError(f"Cannot reshape frame with shape {arr.shape} into {n} x {n}")


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        prefix = ""
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            prefix += chunk
            match = re.search(r'"metadata"\s*:\s*(\{.*?\})\s*,\s*"data"\s*:', prefix, flags=re.S)
            if match:
                return dict(json.loads(match.group(1)))
            if len(prefix) > 8 * 1024 * 1024:
                break
    raise ValueError(f"Could not parse metadata from {path}")


def iter_json_frames(path: Path):
    """Yield frames from the top-level `data` array without loading the file."""
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    with path.open() as handle:
        buffer = ""
        pos = 0
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find data array in {path}")
            buffer += chunk
            match = re.search(r'"data"\s*:\s*\[', buffer, flags=re.S)
            if match:
                pos = match.end()
                break
            if len(buffer) > 8 * chunk_size:
                buffer = buffer[-chunk_size:]

        frame_index = 0
        eof = False
        while True:
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buffer):
                    break
                chunk = handle.read(chunk_size)
                if not chunk:
                    eof = True
                    break
                buffer += chunk
            if eof:
                break
            if buffer[pos] == "]":
                break
            while True:
                try:
                    frame, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk
            yield frame_index, frame
            frame_index += 1
            pos = end
            if pos > 4 * chunk_size:
                buffer = buffer[pos:]
                pos = 0


def count_frames(path: Path) -> int:
    total = 0
    for _frame_index, _frame in iter_json_frames(path):
        total += 1
    return total


def find_membrane_trajectory(system_dir: Path, system_name: str) -> Path:
    candidates = [
        system_dir / "out_simulation" / "mem_traj.json",
        system_dir / f"out_n{system_name}_rigid" / "mem_traj.json",
    ]
    candidates.extend(sorted(system_dir.glob("out*/mem_traj.json")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mem_traj.json found below {system_dir}")


def find_particle_trajectory(system_dir: Path, system_name: str) -> Path | None:
    candidates = [
        system_dir / "out_simulation" / "trajectory.gsd",
        system_dir / f"out_n{system_name}_rigid" / "trajectory.gsd",
    ]
    candidates.extend(sorted(system_dir.glob("out*/trajectory.gsd")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def circular_mean_pbc(values: np.ndarray, length: float) -> float:
    theta = 2.0 * np.pi * ((np.asarray(values, dtype=float) + 0.5 * length) % length) / length
    mean_sin = float(np.mean(np.sin(theta)))
    mean_cos = float(np.mean(np.cos(theta)))
    angle = math.atan2(mean_sin, mean_cos)
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle * length / (2.0 * np.pi) - 0.5 * length)


def get_h_indices(x: np.ndarray, y: np.ndarray, length: float, spacing: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    xind = (((np.asarray(x) + (spacing / 2.0) + (length / 2.0)) % length) / spacing).astype(int)
    yind = (((np.asarray(y) + (spacing / 2.0) + (length / 2.0)) % length) / spacing).astype(int)
    return np.clip(xind, 0, n - 1), np.clip(yind, 0, n - 1)


def load_membrane_bond_info(system_dir: Path, system_name: str) -> dict[str, Any]:
    processed = system_dir / f"processed_{system_name}_stitched_v2.json"
    if not processed.exists():
        return {
            "processed_file": "",
            "bond_count": 0,
            "bond_sites": "",
            "all_com": False,
            "reason": "processed stitched JSON not found",
        }
    with processed.open() as handle:
        obj = json.load(handle)
    record = obj[0] if isinstance(obj, list) else obj
    membrane_bonds = record.get("membrane_bonds", {}).get("bonds", [])
    bond_sites = sorted({str(bond.get("site", "")) for bond in membrane_bonds})
    return {
        "processed_file": str(processed),
        "bond_count": len(membrane_bonds),
        "bond_sites": ";".join(site for site in bond_sites if site),
        "all_com": bool(membrane_bonds) and all(bond.get("site") == "COM" for bond in membrane_bonds),
        "reason": "" if membrane_bonds else "processed stitched JSON has no membrane_bonds.bonds records",
    }


def gag_com_positions_from_frame(
    frame: Any,
    expected_count: int,
    *,
    require_expected_count: bool = False,
) -> np.ndarray:
    type_names = np.asarray(frame.particles.types)
    typeid = np.asarray(frame.particles.typeid, dtype=int)
    names = type_names[typeid]
    mask = names == "A"
    positions = np.asarray(frame.particles.position, dtype=float)[mask]
    if len(positions) == 0:
        raise ValueError("No type-A Gag COM particles found in GSD frame")
    if require_expected_count and len(positions) != expected_count:
        # The current datasets tether every Gag COM. Keep the failure explicit
        # in exact mode because a different site definition needs a real
        # particle-index map instead of a guessed particle type.
        raise ValueError(f"Expected {expected_count} bonded COM particles, found {len(positions)} type-A particles")
    return positions


def grid_from_metadata(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    n = int(metadata["N"])
    length = float(metadata["L"])
    spacing = float(metadata.get("a", length / n))
    coords = -0.5 * length + spacing * np.arange(n, dtype=float)
    x_grid, y_grid = np.meshgrid(coords, coords, indexing="ij")
    return x_grid, y_grid, spacing


def centered_circular_mask(x_grid: np.ndarray, y_grid: np.ndarray, radius: float) -> np.ndarray:
    mid_x = x_grid.shape[0] // 2
    mid_y = x_grid.shape[1] // 2
    center_x = float(x_grid[mid_x, mid_y])
    center_y = float(y_grid[mid_x, mid_y])
    r = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
    return r <= radius


def load_gag_footprint(system_dir: Path, system_name: str) -> dict[str, Any]:
    processed = system_dir / f"processed_{system_name}_stitched_v2.json"
    if not processed.exists():
        return {"center_x": 0.0, "center_y": 0.0, "radius": 30.0, "bind_l": 9.0, "n_gags": float(system_name)}
    with processed.open() as handle:
        obj = json.load(handle)
    record = obj[0] if isinstance(obj, list) else obj
    coords = np.asarray(record.get("coords", []), dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        return {"center_x": 0.0, "center_y": 0.0, "radius": 30.0, "bind_l": 9.0, "n_gags": float(system_name)}

    xy = coords[:, :2]
    center = np.median(xy, axis=0)
    radii = np.sqrt(np.sum((xy - center) ** 2, axis=1))
    membrane_bonds = record.get("membrane_bonds", {})
    bind_l = finite_or_nan(membrane_bonds.get("bind_l", 9.0))
    if not np.isfinite(bind_l):
        bind_l = 9.0
    cap_radius = float(np.nanquantile(radii, 0.95) + bind_l)
    cap_radius = max(cap_radius, 2.0 * bind_l)
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "radius": cap_radius,
        "bind_l": bind_l,
        "n_gags": float(len(coords)),
    }


def graph_mean_curvature(z: np.ndarray, spacing: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(z, spacing, spacing, edge_order=2)
    d2z_dyy, d2z_dxy_from_y = np.gradient(dz_dy, spacing, spacing, edge_order=2)
    d2z_dyx_from_x, d2z_dxx = np.gradient(dz_dx, spacing, spacing, edge_order=2)
    d2z_dxy = 0.5 * (d2z_dxy_from_y + d2z_dyx_from_x)
    denom = 2.0 * np.power(1.0 + dz_dx * dz_dx + dz_dy * dz_dy, 1.5)
    numer = (1.0 + dz_dy * dz_dy) * d2z_dxx
    numer -= 2.0 * dz_dx * dz_dy * d2z_dxy
    numer += (1.0 + dz_dx * dz_dx) * d2z_dyy
    return numer / np.maximum(denom, 1.0e-15)


def surface_weights(z: np.ndarray, spacing: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(z, spacing, spacing, edge_order=2)
    return np.sqrt(1.0 + dz_dx * dz_dx + dz_dy * dz_dy) * spacing * spacing


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan")}
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": float(np.mean(values)),
        "std": std,
        "sem": std / math.sqrt(len(values)) if len(values) > 1 else 0.0,
    }


def weighted_sphere_initial_guess(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if np.sum(weights) <= 0:
        weights = np.ones(len(points), dtype=float)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    matrix = np.column_stack([2.0 * x, 2.0 * y, 2.0 * z, np.ones(len(points))])
    rhs = x * x + y * y + z * z
    scaled_matrix = matrix * np.sqrt(weights[:, None])
    scaled_rhs = rhs * np.sqrt(weights)
    solution, *_ = np.linalg.lstsq(scaled_matrix, scaled_rhs, rcond=None)
    center = solution[:3]
    radius_sq = float(solution[3] + np.dot(center, center))
    radius = float(np.sqrt(radius_sq)) if radius_sq > 0 else SPHERE_RADIUS_NM
    if not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
        center = np.array([0.0, 0.0, -50.0], dtype=float)
        radius = SPHERE_RADIUS_NM
    return center.astype(float), float(np.clip(radius, 5.0, 500.0))


def ellipsoid_geometric_residual(
    parameters: np.ndarray,
    points: np.ndarray,
    sqrt_weights: np.ndarray,
) -> np.ndarray:
    center = parameters[:3]
    axes = np.exp(parameters[3:6])
    shifted = points - center
    scaled = shifted / axes
    implicit = np.sum(scaled * scaled, axis=1) - 1.0
    grad_norm = 2.0 * np.sqrt(np.sum((shifted / (axes * axes)) ** 2, axis=1))
    signed_distance = implicit / np.maximum(grad_norm, 1.0e-10)
    return sqrt_weights * signed_distance


def curvature_radius_metrics_from_axes(axes: np.ndarray) -> dict[str, float]:
    axes = np.asarray(axes, dtype=float)
    if len(axes) != 3 or not np.all(np.isfinite(axes)) or axes[2] <= 0:
        return {key: float("nan") for key in ELLIPSOID_KEYS if "Ellipsoid" in key or "Sphere" in key or "Closest" in key}
    radii = np.array([axes[0] * axes[0] / axes[2], axes[1] * axes[1] / axes[2]], dtype=float)
    radius_delta = radii - SPHERE_RADIUS_NM
    rms_deviation_nm = float(np.sqrt(np.mean(radius_delta * radius_delta)))
    mean_radius = float(np.mean(radii))
    closest_delta = radii - mean_radius
    closest_rms_deviation_nm = float(np.sqrt(np.mean(closest_delta * closest_delta)))
    min_radius = float(np.min(radii))
    max_radius = float(np.max(radii))
    return {
        "EllipsoidApexRadiusXNm": float(radii[0]),
        "EllipsoidApexRadiusYNm": float(radii[1]),
        "EllipsoidApexRadiusMeanNm": mean_radius,
        "EllipsoidApexRadiusSpreadNm": float(np.max(radii) - np.min(radii)),
        "EllipsoidApexRadiusAnisotropyFraction": float((np.max(radii) - np.min(radii)) / mean_radius)
        if mean_radius > 0
        else float("nan"),
        "SphereCurvatureRadiusRmsDeviationNm": rms_deviation_nm,
        "SphereCurvatureRadiusRmsDeviationFraction": rms_deviation_nm / SPHERE_RADIUS_NM,
        "SphereCurvatureRadiusRmsDeviationPercent": 100.0 * rms_deviation_nm / SPHERE_RADIUS_NM,
        "ClosestSphereRadiusNm": mean_radius,
        "ClosestSphereRadiusRmsDeviationNm": closest_rms_deviation_nm,
        "ClosestSphereRadiusRmsDeviationFraction": closest_rms_deviation_nm / mean_radius
        if mean_radius > 0
        else float("nan"),
        "ClosestSphereRadiusRmsDeviationPercent": 100.0 * closest_rms_deviation_nm / mean_radius
        if mean_radius > 0
        else float("nan"),
        "EllipsoidApexEccentricity": float(np.sqrt(max(0.0, 1.0 - (min_radius / max_radius) ** 2)))
        if max_radius > 0
        else float("nan"),
    }


def fit_axis_aligned_ellipsoid(points: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    output = {key: float("nan") for key in ELLIPSOID_KEYS}
    if least_squares is None:
        output["EllipsoidFitConverged"] = False
        return output
    valid = np.all(np.isfinite(points), axis=1) & np.isfinite(weights) & (weights > 0)
    points = points[valid]
    weights = weights[valid]
    if len(points) < 20:
        output["EllipsoidFitConverged"] = False
        return output
    if len(points) > MAX_FIT_SAMPLES:
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        angles = np.arctan2(points[:, 1], points[:, 0])
        order = np.lexsort((angles, radii))
        take = np.unique(np.round(np.linspace(0, len(order) - 1, MAX_FIT_SAMPLES)).astype(int))
        points = points[order[take]]
        weights = weights[order[take]]
    weights = weights / np.nanmean(weights)
    sqrt_weights = np.sqrt(weights)
    center0, radius0 = weighted_sphere_initial_guess(points, weights)
    axes0 = np.full(3, radius0, dtype=float)
    initial = np.concatenate([center0, np.log(axes0)])
    lower = np.array([-150.0, -150.0, -500.0, np.log(5.0), np.log(5.0), np.log(5.0)])
    upper = np.array([150.0, 150.0, 150.0, np.log(500.0), np.log(500.0), np.log(500.0)])
    initial = np.minimum(np.maximum(initial, lower + 1.0e-6), upper - 1.0e-6)
    try:
        fit = least_squares(
            ellipsoid_geometric_residual,
            initial,
            args=(points, sqrt_weights),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.25,
            max_nfev=600,
        )
    except Exception:
        output["EllipsoidFitConverged"] = False
        return output
    center = fit.x[:3]
    axes = np.exp(fit.x[3:6])
    residuals = ellipsoid_geometric_residual(fit.x, points, np.ones_like(sqrt_weights))
    sorted_axes = np.sort(axes)
    axis_delta = axes - SPHERE_RADIUS_NM
    output.update(
        {
            "EllipsoidCenterXNm": float(center[0]),
            "EllipsoidCenterYNm": float(center[1]),
            "EllipsoidCenterZNm": float(center[2]),
            "EllipsoidAxisXNm": float(axes[0]),
            "EllipsoidAxisYNm": float(axes[1]),
            "EllipsoidAxisZNm": float(axes[2]),
            "EllipsoidAxisMinNm": float(sorted_axes[0]),
            "EllipsoidAxisMidNm": float(sorted_axes[1]),
            "EllipsoidAxisMaxNm": float(sorted_axes[2]),
            "EllipsoidAxisMeanNm": float(np.mean(axes)),
            "EllipsoidAxisSpreadNm": float(np.max(axes) - np.min(axes)),
            "SphereAxisRmsDeviationNm": float(np.sqrt(np.mean(axis_delta * axis_delta))),
            "SphereAxisRmsDeviationFraction": float(np.sqrt(np.mean(axis_delta * axis_delta)) / SPHERE_RADIUS_NM),
            "SphereAxisRmsDeviationPercent": float(100.0 * np.sqrt(np.mean(axis_delta * axis_delta)) / SPHERE_RADIUS_NM),
            "SphereAxisMeanAbsDeviationNm": float(np.mean(np.abs(axis_delta))),
            "SphereAxisMaxAbsDeviationNm": float(np.max(np.abs(axis_delta))),
            "EllipsoidFitRmseNm": float(np.sqrt(np.average(residuals * residuals, weights=weights))),
            "EllipsoidFitMedianAbsResidualNm": float(np.median(np.abs(residuals))),
            "EllipsoidFitConverged": bool(fit.success),
            "EllipsoidFitCost": float(fit.cost),
        }
    )
    output.update(curvature_radius_metrics_from_axes(axes))
    return output


def fit_quadratic_curvature_patch(points: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Fit a local graph patch and estimate principal curvature radii.

    A full ellipsoid is weakly constrained for nearly flat open caps.  This
    local quadratic fit gives the sphere-deviation analogue directly from the
    fitted cap curvature at the footprint center.
    """
    output = {key: float("nan") for key in QUADRATIC_KEYS}
    valid = np.all(np.isfinite(points), axis=1) & np.isfinite(weights) & (weights > 0)
    points = points[valid]
    weights = weights[valid]
    if len(points) < 20:
        return output
    weights = weights / np.nanmean(weights)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    center_x = float(np.average(x, weights=weights))
    center_y = float(np.average(y, weights=weights))
    u = x - center_x
    v = y - center_y
    design = np.column_stack([np.ones_like(u), u, v, u * u, u * v, v * v])
    weighted_design = design * np.sqrt(weights[:, None])
    weighted_z = z * np.sqrt(weights)
    try:
        coef, *_ = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)
    except np.linalg.LinAlgError:
        return output
    residuals = z - design @ coef
    zx = float(coef[1])
    zy = float(coef[2])
    zxx = float(2.0 * coef[3])
    zxy = float(coef[4])
    zyy = float(2.0 * coef[5])
    normalizer = math.sqrt(1.0 + zx * zx + zy * zy)
    first_form = np.array([[1.0 + zx * zx, zx * zy], [zx * zy, 1.0 + zy * zy]], dtype=float)
    second_form = np.array([[zxx, zxy], [zxy, zyy]], dtype=float) / normalizer
    try:
        shape_operator = np.linalg.solve(first_form, second_form)
        curvatures = np.linalg.eigvals(shape_operator).real
    except np.linalg.LinAlgError:
        return output
    abs_curvatures = np.sort(np.abs(curvatures))
    radii = 1.0 / np.maximum(abs_curvatures, 1.0e-12)
    mean_radius = float(np.mean(radii))
    closest_delta = radii - mean_radius
    closest_rms = float(np.sqrt(np.mean(closest_delta * closest_delta)))
    sphere50_delta = radii - SPHERE_RADIUS_NM
    sphere50_rms = float(np.sqrt(np.mean(sphere50_delta * sphere50_delta)))
    min_radius = float(np.min(radii))
    max_radius = float(np.max(radii))
    output.update(
        {
            "QuadraticFitRmseNm": float(np.sqrt(np.average(residuals * residuals, weights=weights))),
            "QuadraticPrincipalCurvature1PerNm": float(curvatures[0]),
            "QuadraticPrincipalCurvature2PerNm": float(curvatures[1]),
            "QuadraticApexRadius1Nm": float(radii[0]),
            "QuadraticApexRadius2Nm": float(radii[1]),
            "QuadraticClosestSphereRadiusNm": mean_radius,
            "QuadraticClosestSphereRadiusRmsDeviationNm": closest_rms,
            "QuadraticClosestSphereRadiusRmsDeviationPercent": 100.0 * closest_rms / mean_radius
            if mean_radius > 0
            else float("nan"),
            "QuadraticSphere50RadiusRmsDeviationNm": sphere50_rms,
            "QuadraticSphere50RadiusRmsDeviationPercent": 100.0 * sphere50_rms / SPHERE_RADIUS_NM,
            "QuadraticApexEccentricity": float(np.sqrt(max(0.0, 1.0 - (min_radius / max_radius) ** 2)))
            if max_radius > 0
            else float("nan"),
        }
    )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_system(system_dir: Path, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    system_name = system_dir.name
    traj = find_membrane_trajectory(system_dir, system_name)
    particle_traj = find_particle_trajectory(system_dir, system_name)
    print(f"[{system_name}] reading metadata {traj}", flush=True)
    metadata = load_metadata(traj)
    x_grid, y_grid, spacing = grid_from_metadata(metadata)
    length = float(metadata["L"])
    grid_n = int(metadata["N"])
    total_frames = count_frames(traj)
    start_frame = total_frames // 2
    print(f"[{system_name}] processing frames {start_frame}:{total_frames}", flush=True)
    footprint = load_gag_footprint(system_dir, system_name)
    expected_gags = int(footprint["n_gags"])
    bond_info = load_membrane_bond_info(system_dir, system_name)
    r = np.sqrt((x_grid - footprint["center_x"]) ** 2 + (y_grid - footprint["center_y"]) ** 2)
    fallback_cap_mask = r <= footprint["radius"]
    centered_fallback_cap_mask = centered_circular_mask(x_grid, y_grid, float(footprint["radius"]))
    interior_mask = np.ones_like(fallback_cap_mask, dtype=bool)
    interior_mask[[0, -1], :] = False
    interior_mask[:, [0, -1]] = False
    use_particle_centering = particle_traj is not None and gsd_hoomd is not None
    use_exact_attachment_mask = use_particle_centering and bool(bond_info["all_com"])
    cap_mask_mode = "gsd_centered_exact_attached_nodes" if use_exact_attachment_mask else "fallback_projected_gag_footprint"
    cap_mask_fallback_reason = ""
    if not use_exact_attachment_mask:
        if particle_traj is None:
            cap_mask_fallback_reason = "trajectory.gsd not found; using uncentered projected Gag-footprint mask"
        elif gsd_hoomd is None:
            cap_mask_fallback_reason = "Python gsd.hoomd unavailable; using uncentered projected Gag-footprint mask"
        elif not bond_info["all_com"]:
            cap_mask_mode = "gsd_centered_projected_gag_footprint"
            cap_mask_fallback_reason = bond_info["reason"] or (
                f"membrane_bonds sites are not all COM ({bond_info['bond_sites']}); "
                "using Gag-centered projected footprint mask"
            )
    particle_frames = None
    if use_particle_centering:
        particle_frames = gsd_hoomd.open(str(particle_traj), mode="r")
        if len(particle_frames) < total_frames:
            raise ValueError(f"{particle_traj} has {len(particle_frames)} frames but membrane trajectory has {total_frames}")
        if use_exact_attachment_mask:
            print(f"[{system_name}] using GSD-centered exact attachment cap mask from {particle_traj}", flush=True)
        else:
            print(
                f"[{system_name}] using GSD-centered projected footprint cap mask from {particle_traj}: "
                f"{cap_mask_fallback_reason}",
                flush=True,
            )
    else:
        print(f"[{system_name}] falling back to uncentered projected Gag-footprint cap mask: {cap_mask_fallback_reason}", flush=True)

    z_mean = np.zeros_like(x_grid, dtype=np.float64)
    z_m2 = np.zeros_like(x_grid, dtype=np.float64)
    cap_attachment_count = np.zeros_like(x_grid, dtype=np.int64)
    cap_attachment_multiplicity = np.zeros_like(x_grid, dtype=np.int64)
    n_frames_used = 0
    frame_rows: list[dict[str, Any]] = []
    height_max_values = []
    height_mean_values = []
    cap_tip_values = []
    cap_mean_values = []
    non_cap_mean_values = []
    curvature_values = []
    unique_attached_counts = []
    center_x_values = []
    center_y_values = []
    center_shift_x_values = []
    center_shift_y_values = []

    for frame_index, frame in iter_json_frames(traj):
        if frame_index < start_frame:
            continue
        z = frame_to_grid(frame, grid_n)
        center_x = float("nan")
        center_y = float("nan")
        shift_x = 0
        shift_y = 0
        if use_particle_centering:
            particle_frame = particle_frames[frame_index]
            gag_positions = gag_com_positions_from_frame(
                particle_frame,
                expected_gags,
                require_expected_count=use_exact_attachment_mask,
            )
            center_x = circular_mean_pbc(gag_positions[:, 0], length)
            center_y = circular_mean_pbc(gag_positions[:, 1], length)
            center_ix, center_iy = get_h_indices(np.array([center_x]), np.array([center_y]), length, spacing, grid_n)
            mid_ind = grid_n // 2
            shift_x = int(mid_ind - center_ix[0])
            shift_y = int(mid_ind - center_iy[0])
            z = np.roll(z, shift_x, axis=0)
            z = np.roll(z, shift_y, axis=1)
            if use_exact_attachment_mask:
                attach_ix, attach_iy = get_h_indices(gag_positions[:, 0], gag_positions[:, 1], length, spacing, grid_n)
                attach_ix = (attach_ix + shift_x) % grid_n
                attach_iy = (attach_iy + shift_y) % grid_n
                frame_cap_mask = np.zeros_like(fallback_cap_mask, dtype=bool)
                frame_cap_multiplicity = np.zeros_like(cap_attachment_multiplicity, dtype=np.int64)
                np.add.at(frame_cap_multiplicity, (attach_ix, attach_iy), 1)
                frame_cap_mask = frame_cap_multiplicity > 0
            else:
                frame_cap_mask = centered_fallback_cap_mask
                frame_cap_multiplicity = frame_cap_mask.astype(np.int64)
        else:
            frame_cap_mask = fallback_cap_mask
            frame_cap_multiplicity = frame_cap_mask.astype(np.int64)
        frame_non_cap_mask = ~frame_cap_mask
        frame_cap_interior_mask = frame_cap_mask & interior_mask

        n_frames_used += 1
        delta = z - z_mean
        z_mean += delta / n_frames_used
        z_m2 += delta * (z - z_mean)
        cap_attachment_count += frame_cap_mask.astype(np.int64)
        cap_attachment_multiplicity += frame_cap_multiplicity

        cap_tip = float(np.max(z[frame_cap_mask]))
        cap_mean = float(np.mean(z[frame_cap_mask]))
        non_cap_mean = float(np.mean(z[frame_non_cap_mask]))
        height_max = cap_tip - non_cap_mean
        height_mean = cap_mean - non_cap_mean
        curvature = graph_mean_curvature(z, spacing)
        mean_curvature = float(np.mean(np.abs(curvature[frame_cap_interior_mask])))
        cap_tip_values.append(cap_tip)
        cap_mean_values.append(cap_mean)
        non_cap_mean_values.append(non_cap_mean)
        height_max_values.append(height_max)
        height_mean_values.append(height_mean)
        curvature_values.append(mean_curvature)
        unique_attached_counts.append(int(np.sum(frame_cap_mask)))
        center_x_values.append(center_x)
        center_y_values.append(center_y)
        center_shift_x_values.append(shift_x)
        center_shift_y_values.append(shift_y)
        frame_rows.append(
            {
                "System": system_name,
                "FrameIndex": frame_index,
                "GagCenterXNm": center_x,
                "GagCenterYNm": center_y,
                "CenterShiftGridX": shift_x,
                "CenterShiftGridY": shift_y,
                "UniqueAttachedNodeCount": int(np.sum(frame_cap_mask)),
                "AttachedNodeMultiplicityCount": int(np.sum(frame_cap_multiplicity)),
                "CapTipMaxZNm": cap_tip,
                "CapMeanZNm": cap_mean,
                "NonCapMeanZNm": non_cap_mean,
                "CapHeightMaxNm": height_max,
                "CapHeightMeanNm": height_mean,
                "MeanAbsCurvatureCapPerNm": mean_curvature,
            }
        )

    if n_frames_used == 0:
        raise ValueError(f"No canonical frames processed for {system_name} from {traj}")
    z_std = np.sqrt(z_m2 / (n_frames_used - 1)) if n_frames_used > 1 else np.zeros_like(z_mean)
    z_sem = z_std / math.sqrt(n_frames_used) if n_frames_used > 1 else np.zeros_like(z_mean)
    n_x, n_y = z_mean.shape
    cap_frequency = cap_attachment_count / n_frames_used
    if use_exact_attachment_mask:
        cap_mask = cap_frequency > 0
    else:
        cap_mask = centered_fallback_cap_mask if use_particle_centering else fallback_cap_mask
        cap_frequency = cap_mask.astype(float)
    cap_interior_mask = cap_mask & interior_mask
    non_cap_mask = ~cap_mask

    node_rows: list[dict[str, Any]] = []
    for ix in range(n_x):
        for iy in range(n_y):
            node_index = ix * n_y + iy
            node_rows.append(
                {
                    "System": system_name,
                    "NodeIndex": node_index,
                    "Ix": ix,
                    "Iy": iy,
                    "XMeanNm": float(x_grid[ix, iy]),
                    "YMeanNm": float(y_grid[ix, iy]),
                    "ZMeanNm": float(z_mean[ix, iy]),
                    "XStdNm": 0.0,
                    "YStdNm": 0.0,
                    "ZStdNm": float(z_std[ix, iy]),
                    "ZSemNm": float(z_sem[ix, iy]),
                    "InCap": bool(cap_mask[ix, iy]),
                    "CapAttachmentFrequency": float(cap_frequency[ix, iy]),
                    "CapAttachmentMultiplicity": int(cap_attachment_multiplicity[ix, iy]),
                }
            )

    mean_non_cap_mean = float(np.mean(z_mean[non_cap_mask]))
    mean_cap_tip = float(np.max(z_mean[cap_mask]))
    mean_cap_mean = float(np.mean(z_mean[cap_mask]))
    mean_curvature_grid = graph_mean_curvature(z_mean, spacing)
    mean_surface_weights = surface_weights(z_mean, spacing)
    fit_points = np.column_stack([x_grid[cap_interior_mask], y_grid[cap_interior_mask], z_mean[cap_interior_mask]])
    fit_weights = mean_surface_weights[cap_interior_mask]
    ellipsoid_metrics = fit_axis_aligned_ellipsoid(fit_points, fit_weights)
    quadratic_metrics = fit_quadratic_curvature_patch(fit_points, fit_weights)

    height_max_summary = summarize(np.asarray(height_max_values))
    height_mean_summary = summarize(np.asarray(height_mean_values))
    cap_tip_summary = summarize(np.asarray(cap_tip_values))
    cap_mean_summary = summarize(np.asarray(cap_mean_values))
    non_cap_mean_summary = summarize(np.asarray(non_cap_mean_values))
    curvature_summary = summarize(np.asarray(curvature_values))
    unique_attached_summary = summarize(np.asarray(unique_attached_counts))
    center_x_summary = summarize(np.asarray(center_x_values))
    center_y_summary = summarize(np.asarray(center_y_values))
    shift_x_summary = summarize(np.asarray(center_shift_x_values))
    shift_y_summary = summarize(np.asarray(center_shift_y_values))
    summary = {
        "System": system_name,
        "Ngags": int(footprint["n_gags"]),
        "TrajectoryFile": str(traj),
        "ParticleTrajectoryFile": str(particle_traj) if particle_traj is not None else "",
        "CapMaskMode": cap_mask_mode,
        "CapMaskFallbackReason": cap_mask_fallback_reason,
        "MembraneBondProcessedFile": bond_info["processed_file"],
        "MembraneBondRecordCount": int(bond_info["bond_count"]),
        "MembraneBondSites": bond_info["bond_sites"],
        "GSDHoomdAvailable": bool(gsd_hoomd is not None),
        "ParticleCenteringEnabled": bool(use_particle_centering),
        "TotalFrames": int(total_frames),
        "CanonicalStartFrame": int(start_frame),
        "CanonicalFrames": int(n_frames_used),
        "GridN": int(n_x),
        "GridSpacingNm": float(spacing),
        "CapCenterXNm": float(footprint["center_x"]),
        "CapCenterYNm": float(footprint["center_y"]),
        "CapRadiusNm": float(footprint["radius"]),
        "CapNodeCount": int(np.sum(cap_mask)),
        "CapInteriorNodeCount": int(np.sum(cap_interior_mask)),
        "FrameUniqueAttachedNodeCountMean": unique_attached_summary["mean"],
        "FrameUniqueAttachedNodeCountStd": unique_attached_summary["std"],
        "FrameUniqueAttachedNodeCountSem": unique_attached_summary["sem"],
        "GagCenterXMeanNm": center_x_summary["mean"],
        "GagCenterXStdNm": center_x_summary["std"],
        "GagCenterYMeanNm": center_y_summary["mean"],
        "GagCenterYStdNm": center_y_summary["std"],
        "CenterShiftGridXMean": shift_x_summary["mean"],
        "CenterShiftGridXStd": shift_x_summary["std"],
        "CenterShiftGridYMean": shift_y_summary["mean"],
        "CenterShiftGridYStd": shift_y_summary["std"],
        "MeanSurfaceNonCapMeanZNm": mean_non_cap_mean,
        "MeanSurfaceCapTipMaxZNm": mean_cap_tip,
        "MeanSurfaceCapMeanZNm": mean_cap_mean,
        "MeanSurfaceCapHeightMaxNm": mean_cap_tip - mean_non_cap_mean,
        "MeanSurfaceCapHeightMeanNm": mean_cap_mean - mean_non_cap_mean,
        "MeanSurfaceMeanAbsCurvatureCapPerNm": float(np.mean(np.abs(mean_curvature_grid[cap_interior_mask]))),
        "FrameCapTipMaxMeanZNm": cap_tip_summary["mean"],
        "FrameCapTipMaxStdZNm": cap_tip_summary["std"],
        "FrameCapTipMaxSemZNm": cap_tip_summary["sem"],
        "FrameCapMeanMeanZNm": cap_mean_summary["mean"],
        "FrameCapMeanStdZNm": cap_mean_summary["std"],
        "FrameCapMeanSemZNm": cap_mean_summary["sem"],
        "FrameNonCapMeanMeanZNm": non_cap_mean_summary["mean"],
        "FrameNonCapMeanStdZNm": non_cap_mean_summary["std"],
        "FrameNonCapMeanSemZNm": non_cap_mean_summary["sem"],
        "FrameCapHeightMaxMeanNm": height_max_summary["mean"],
        "FrameCapHeightMaxStdNm": height_max_summary["std"],
        "FrameCapHeightMaxSemNm": height_max_summary["sem"],
        "FrameCapHeightMeanMeanNm": height_mean_summary["mean"],
        "FrameCapHeightMeanStdNm": height_mean_summary["std"],
        "FrameCapHeightMeanSemNm": height_mean_summary["sem"],
        "FrameMeanAbsCurvatureCapMeanPerNm": curvature_summary["mean"],
        "FrameMeanAbsCurvatureCapStdPerNm": curvature_summary["std"],
        "FrameMeanAbsCurvatureCapSemPerNm": curvature_summary["sem"],
    }
    summary.update(ellipsoid_metrics)
    summary.update(quadratic_metrics)

    write_csv(output_dir / f"{system_name}_canonical_node_stats.csv", node_rows)
    write_csv(output_dir / f"{system_name}_second_half_frame_features.csv", frame_rows)
    np.savez_compressed(
        output_dir / f"{system_name}_canonical_surface_arrays.npz",
        x=x_grid,
        y=y_grid,
        z_mean=z_mean,
        z_std=z_std,
        z_sem=z_sem,
        cap_mask=cap_mask,
        cap_attachment_frequency=cap_frequency,
        cap_attachment_multiplicity=cap_attachment_multiplicity,
        mean_curvature=mean_curvature_grid,
    )
    print(f"[{system_name}] done: {n_frames_used} second-half frames, {int(np.sum(cap_mask))} cap nodes", flush=True)
    return node_rows, frame_rows, summary


def maybe_make_plots(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    systems = [row["System"] for row in summary_rows]
    x = np.arange(len(systems))
    metrics = [
        ("FrameCapHeightMaxMeanNm", "Cap height (nm)", "cap_height", "FrameCapHeightMaxStdNm"),
        (
            "MeanSurfaceMeanAbsCurvatureCapPerNm",
            "Mean curvature of cap (nm$^{-1}$)",
            "mean_curvature_cap",
            None,
        ),
        ("ClosestSphereRadiusRmsDeviationPercent", "Closest-sphere deviation (%)", "closest_sphere_deviation", None),
        (
            "QuadraticClosestSphereRadiusRmsDeviationPercent",
            "Quadratic closest-sphere deviation (%)",
            "quadratic_closest_sphere_deviation",
            None,
        ),
        ("EllipsoidApexEccentricity", "Ellipsoid apex eccentricity", "ellipsoid_apex_eccentricity", None),
        ("QuadraticApexEccentricity", "Quadratic apex eccentricity", "quadratic_apex_eccentricity", None),
    ]
    for col, ylabel, stem, err_col in metrics:
        y = np.asarray([finite_or_nan(row.get(col)) for row in summary_rows], dtype=float)
        yerr = None
        if err_col is not None:
            yerr = np.asarray([finite_or_nan(row.get(err_col)) for row in summary_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.bar(x, y, yerr=yerr, color="0.35", width=0.65, capsize=4)
        ax.set_xticks(x, systems)
        ax.set_xlabel("Gag count")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}.png", dpi=300)
        fig.savefig(output_dir / f"{stem}.svg")
        plt.close(fig)


def write_readme(output_dir: Path) -> None:
    readme = output_dir / "README.md"
    readme.write_text(
        """# HOOMD k250 Canonical Geometry Analysis

This folder was generated by `analyze_hoomd_k250_geometry.py`.

Definitions:

- Source trajectory: `out_simulation/mem_traj.json` or `out_n*_rigid/mem_traj.json`
  in each system folder.  When `trajectory.gsd` is available and readable, it is
  used to reconstruct the tethered Gag COM positions for cap masking and
  centering.
- Membrane representation: an `N x N` height field. The VTF convention is used:
  node `(ix, iy)` has `x = -L/2 + ix * a`, `y = -L/2 + iy * a`, and sampled
  `z` from the JSON trajectory.
- Canonical window: the second half of each trajectory, `frames[n_frames//2:]`.
- Node statistics: x/y are the centered membrane grid coordinates; z
  mean/std/SEM are computed across the canonical window after any Gag-centering
  roll.
- Gag cap mask: primary mode is `gsd_centered_exact_attached_nodes`. For every
  canonical-window frame, the script reads the Gag particle trajectory, computes
  a periodic-boundary circular mean Gag center, rolls the membrane height field
  so that center lands on the middle grid node, reconstructs the membrane grid
  nodes directly tethered to Gag COM particles using the same `get_h_indices`
  convention as the HOOMD setup, and uses those nodes as the cap mask. The
  processed JSON records the bonded Gag sites, but in the current files it does
  not store explicit membrane-node IDs; those membrane nodes are reconstructed
  from the GSD COM positions. The summary CSV records `CapMaskMode`,
  `ParticleCenteringEnabled`, `MembraneBondRecordCount`, `MembraneBondSites`,
  and `CapMaskFallbackReason` so this can be audited.
- Fallback cap mask: if `trajectory.gsd` and `gsd.hoomd` are available but the
  bond metadata is not compatible with exact COM attachment reconstruction, the
  script still centers each membrane frame on the PBC-aware Gag COM and then
  applies the projected Gag-footprint radius mask in that centered frame
  (`gsd_centered_projected_gag_footprint`). If GSD is unavailable, the script
  uses the older uncentered projected footprint mask and records that reason in
  the summary.
- Cap height: for each canonical-window frame, estimate the resting membrane
  height as the mean z over non-cap nodes, estimate the cap tip as the maximum
  z over cap nodes, and compute `CapHeightMaxNm = CapTipMaxZNm -
  NonCapMeanZNm`.  The reported canonical cap height is the mean of that
  frame-wise quantity.  The summary also records the two ingredients as
  `FrameCapTipMaxMeanZNm` and `FrameNonCapMeanMeanZNm`.
- Mean curvature of cap: graph mean curvature of `z(x,y)`, averaged as absolute
  curvature over interior cap nodes.
- Ellipsoid/sphere metrics: an axis-aligned ellipsoid is fit to the canonical
  mean cap surface with area-element weights. The closest-sphere deviation is
  the RMS deviation of the ellipsoid apex curvature radii from their own mean,
  reported as a percent of that closest radius. The fixed-sphere deviation uses
  a 50 nm reference radius.
- Quadratic curvature metrics: because a full ellipsoid can be weakly
  constrained for shallow open caps, the script also fits a weighted local
  quadratic graph patch to the canonical mean cap. Its principal curvature radii
  define a robust closest-sphere deviation and apex eccentricity.

Key outputs:

- `canonical_surface_summary.csv`: one row per Gag system.
- `canonical_node_stats.csv`: concatenated per-node canonical mean/std locations.
- `second_half_frame_features.csv`: per-frame cap height and curvature over the
  canonical window.
- `*_canonical_surface_arrays.npz`: x/y, mean z, z std/SEM, cap mask,
  cap-attachment frequency/multiplicity, and curvature arrays per system.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("analysis_hoomd_geometry/results"))
    parser.add_argument("--systems", nargs="*", default=["203", "252", "408", "454"])
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_node_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for system in args.systems:
        system_dir = root / system
        node_rows, frame_rows, summary = analyze_system(system_dir, output_dir)
        all_node_rows.extend(node_rows)
        all_frame_rows.extend(frame_rows)
        summary_rows.append(summary)

    write_csv(output_dir / "canonical_node_stats.csv", all_node_rows)
    write_csv(output_dir / "second_half_frame_features.csv", all_frame_rows)
    write_csv(output_dir / "canonical_surface_summary.csv", summary_rows)
    maybe_make_plots(output_dir, summary_rows)
    write_readme(output_dir)
    print(f"Wrote analysis outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
