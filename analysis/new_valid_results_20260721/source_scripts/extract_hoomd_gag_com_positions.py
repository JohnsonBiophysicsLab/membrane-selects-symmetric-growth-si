#!/usr/bin/env python3
"""Extract canonical-window Gag COM positions from HOOMD GSD trajectories.

Run this on Rockfish where `gsd.hoomd` is available and the source
`trajectory.gsd` files live. The output CSV is intentionally small and can be
rsynced back for plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

import gsd.hoomd


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
            if eof or buffer[pos] == "]":
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
    return sum(1 for _frame_index, _frame in iter_json_frames(path))


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


def wrap_box(values: np.ndarray, length: float) -> np.ndarray:
    return ((np.asarray(values, dtype=float) + 0.5 * length) % length) - 0.5 * length


def find_membrane_trajectory(system_dir: Path, system: str) -> Path:
    candidates = [
        system_dir / "out_simulation" / "mem_traj.json",
        system_dir / f"out_n{system}_rigid" / "mem_traj.json",
    ]
    candidates.extend(sorted(system_dir.glob("out*/mem_traj.json")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mem_traj.json found below {system_dir}")


def find_particle_trajectory(system_dir: Path, system: str) -> Path:
    candidates = [
        system_dir / "out_simulation" / "trajectory.gsd",
        system_dir / f"out_n{system}_rigid" / "trajectory.gsd",
    ]
    candidates.extend(sorted(system_dir.glob("out*/trajectory.gsd")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No trajectory.gsd found below {system_dir}")


def type_a_positions(frame: Any) -> np.ndarray:
    type_names = np.asarray(frame.particles.types)
    typeid = np.asarray(frame.particles.typeid, dtype=int)
    names = type_names[typeid]
    positions = np.asarray(frame.particles.position, dtype=float)[names == "A"]
    if len(positions) == 0:
        raise ValueError("No type-A Gag COM particles found in GSD frame")
    return positions


def analyze_system(root: Path, condition_key: str, condition_label: str, system: str) -> list[dict[str, Any]]:
    system_dir = root / system
    membrane_traj = find_membrane_trajectory(system_dir, system)
    metadata = load_metadata(membrane_traj)
    length = float(metadata["L"])
    n = int(metadata["N"])
    spacing = float(metadata.get("a", length / n))
    mid_ind = n // 2

    gsd_path = find_particle_trajectory(system_dir, system)
    trajectory = gsd.hoomd.open(str(gsd_path), mode="r")
    total_frames = count_frames(membrane_traj)
    if len(trajectory) < total_frames:
        raise ValueError(f"{gsd_path} has {len(trajectory)} frames but {membrane_traj} has {total_frames}")
    start_frame = total_frames // 2
    expected_frames = total_frames - start_frame
    if expected_frames <= 0:
        raise ValueError(f"No canonical frames in {gsd_path}")

    first_positions = type_a_positions(trajectory[start_frame])
    n_gags = len(first_positions)
    sums = np.zeros((n_gags, 3), dtype=float)
    sums2 = np.zeros((n_gags, 3), dtype=float)
    delta_sums = np.zeros(n_gags, dtype=float)
    delta_sums2 = np.zeros(n_gags, dtype=float)
    n_used = 0
    for frame_index, json_frame in iter_json_frames(membrane_traj):
        if frame_index < start_frame:
            continue
        positions = type_a_positions(trajectory[frame_index])
        if len(positions) != n_gags:
            raise ValueError(f"{gsd_path} frame {frame_index} has {len(positions)} type-A particles; expected {n_gags}")
        z_frame = np.asarray(json_frame, dtype=float)
        if z_frame.shape != (n, n):
            z_frame = z_frame.reshape((n, n))
        center_x = circular_mean_pbc(positions[:, 0], length)
        center_y = circular_mean_pbc(positions[:, 1], length)
        center_ix, center_iy = get_h_indices(np.array([center_x]), np.array([center_y]), length, spacing, n)
        shift_x = int(mid_ind - center_ix[0])
        shift_y = int(mid_ind - center_iy[0])
        z_frame = np.roll(z_frame, shift_x, axis=0)
        z_frame = np.roll(z_frame, shift_y, axis=1)
        centered = positions.copy()
        centered[:, 0] = wrap_box(centered[:, 0] + shift_x * spacing, length)
        centered[:, 1] = wrap_box(centered[:, 1] + shift_y * spacing, length)
        attach_ix, attach_iy = get_h_indices(positions[:, 0], positions[:, 1], length, spacing, n)
        attach_ix = (attach_ix + shift_x) % n
        attach_iy = (attach_iy + shift_y) % n
        gag_minus_membrane = centered[:, 2] - z_frame[attach_ix, attach_iy]
        sums += centered
        sums2 += centered * centered
        delta_sums += gag_minus_membrane
        delta_sums2 += gag_minus_membrane * gag_minus_membrane
        n_used += 1

    if n_used != expected_frames:
        raise ValueError(f"Processed {n_used} canonical frames for {system_dir}; expected {expected_frames}")

    means = sums / n_used
    variances = np.maximum(sums2 / n_used - means * means, 0.0)
    stds = np.sqrt(variances)
    sems = stds / math.sqrt(n_used)
    delta_means = delta_sums / n_used
    delta_variances = np.maximum(delta_sums2 / n_used - delta_means * delta_means, 0.0)
    delta_stds = np.sqrt(delta_variances)
    delta_sems = delta_stds / math.sqrt(n_used)
    rows = []
    for gag_index in range(n_gags):
        rows.append(
            {
                "ConditionKey": condition_key,
                "Condition": condition_label,
                "System": system,
                "GagIndex": gag_index,
                "SourceTrajectory": str(gsd_path),
                "CanonicalStartFrame": start_frame,
                "CanonicalFrames": n_used,
                "XMeanNm": means[gag_index, 0],
                "YMeanNm": means[gag_index, 1],
                "ZMeanNm": means[gag_index, 2],
                "XStdNm": stds[gag_index, 0],
                "YStdNm": stds[gag_index, 1],
                "ZStdNm": stds[gag_index, 2],
                "XSemNm": sems[gag_index, 0],
                "YSemNm": sems[gag_index, 1],
                "ZSemNm": sems[gag_index, 2],
                "GagMinusNearestMembraneZMeanNm": delta_means[gag_index],
                "GagMinusNearestMembraneZStdNm": delta_stds[gag_index],
                "GagMinusNearestMembraneZSemNm": delta_sems[gag_index],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsbd-root", type=Path, default=Path("/home/yying7/scr16-mjohn218/fsbd_gag_sims"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--systems", nargs="*", default=["203", "252", "408", "454"])
    args = parser.parse_args()

    conditions = [
        ("k250", "k = 250", args.fsbd_root / "k250"),
        ("rigid", "rigid", args.fsbd_root / "rigid"),
    ]
    rows: list[dict[str, Any]] = []
    for condition_key, condition_label, root in conditions:
        for system in args.systems:
            print(f"[{condition_key} {system}] extracting Gag COM positions", flush=True)
            rows.extend(analyze_system(root, condition_key, condition_label, system))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
