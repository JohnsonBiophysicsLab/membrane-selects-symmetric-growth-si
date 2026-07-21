# Cylindrical Asymmetry Metric

This note documents how `plot_cylindrical_symmetry_rigidity.py` defines the cylindrical-asymmetry metric and its standard error for the rigidity comparison between `a30_b30` and `a30_b15` simulations.

## Purpose

The metric measures how close the membrane cap surface is to a cylindrically symmetric surface of revolution about the vertical axis. It is intended to compare the membrane shapes selected by the minimum total-energy trajectory at each pair of:

- eccentricity `e = 0` for `a30_b30`
- eccentricity `e = 0.866` for `a30_b15`
- protein lattice rigidity `K = 10, 31.6, 100, 316, 1000`

The metric is computed from the membrane surface, not directly from the raw control vertices.

## Source Data

For each trajectory, the script uses:

- `output.log`-derived `Min_Etot` and `Min_step`
- the `vertex*.csv` snapshot nearest to `Min_step`
- the corresponding `face.csv`

For `a30_b30`, source rows come from:

`/Users/yueying/Downloads/change_stiffness/results/rigidity_results.csv`

For `a30_b15`, source rows come from:

`/Users/yueying/Workspace/SLIMED/analysis/cluster_results_20260602/results/all_runs.csv`

For the `a30_b15`, `K = 1000` point, the script prefers the fetched continuation rows from:

`/Users/yueying/Workspace/SLIMED/analysis/cluster_results_20260602/fetched/idealized_lattice/change_stiffness_a30b15_restart_20260528/s1000_continue_20260609`

The full trajectory-level audit table is saved to:

`/Users/yueying/Workspace/SLIMED/analysis/cluster_results_20260602/results/cylindrical_symmetry_candidate_trajectories.csv`

The selected minimum-energy summary table is saved to:

`/Users/yueying/Workspace/SLIMED/analysis/cluster_results_20260602/results/cylindrical_symmetry_surface_summary.csv`

## Surface Construction

The input membrane mesh is treated as the control mesh for a Loop subdivision surface.

For each trajectory:

1. The control vertices are centered in `x` and `y` by subtracting the mean projected vertex position.
2. The lowest `z` coordinate is shifted to zero.
3. The triangular control mesh is refined by three iterations of Loop subdivision.
4. The refined triangular faces are used as surface samples.
5. Each sample is represented by its face centroid and weighted by its refined triangle area.

This means the metric samples the approximated subdivision surface through refined area elements, instead of treating the original coarse vertices as equally weighted data points.

## Boundary Exclusion

The simulation box boundary is square, so circular annuli near the edge would be incompletely represented. Those incomplete annuli can artificially increase or decrease apparent cylindrical symmetry.

To reduce that boundary artifact, the script estimates the largest complete radial region contained inside the centered square-like footprint:

```text
complete_annulus_radius = min(|x_min|, |x_max|, |y_min|, |y_max|) - 2 nm
usable_radius = min(30 nm, complete_annulus_radius)
```

Only refined face centroids with projected radius `r <= usable_radius` are included.

## Metric Definition

For each refined face centroid `i`, define:

```text
r_i = sqrt(x_i^2 + y_i^2)
z_i = centroid height
w_i = refined triangle area
```

The cap is divided into 24 radial bins between `r = 0` and the cap radius. In each radial bin `b`, the area-weighted radial mean height is:

```text
zbar_b = sum_i(w_i z_i) / sum_i(w_i), for samples i in bin b
```

A perfectly cylindrically symmetric surface would have height depending only on `r`, so all samples in the same radial bin would have the same height. The residual from cylindrical symmetry is therefore:

```text
residual_i = z_i - zbar_b
```

The area-weighted RMS residual is:

```text
RMS_residual = sqrt( sum_i(w_i residual_i^2) / sum_i(w_i) )
```

To make the metric dimensionless and comparable across caps, the residual is normalized by the area-weighted robust cap height range:

```text
z_scale = weighted_percentile(z, 95%) - weighted_percentile(z, 5%)
```

The final metric is:

```text
CylindricalAsymmetry = RMS_residual / z_scale
```

Lower values mean the cap is closer to cylindrical symmetry. Higher values mean that at fixed radius the surface height varies more strongly with angle.

## Standard Error

The standard error follows the same joint-resampling convention used for quantities associated with the minimum total-energy state.

For each `(eccentricity, rigidity)` group:

1. Calculate `CylindricalAsymmetry` once for every trajectory row.
2. Bootstrap-resample whole trajectory rows with replacement.
3. In each bootstrap sample, choose the row with the smallest `Min_Etot`.
4. Record the `CylindricalAsymmetry` from that same selected row.
5. Repeat for 10,000 bootstrap samples.
6. Report the standard deviation of the bootstrap-selected asymmetry values as `StdErrCylindricalAsymmetry`.

This is not a bootstrap over mesh vertices, refined faces, or radial bins. It is a joint row-wise bootstrap with respect to the minimum total-energy selection, so the error bar reflects uncertainty in which trajectory would be selected as the minimum-energy representative.

## Plot

The plot is saved as:

`/Users/yueying/Workspace/SLIMED/analysis/cluster_results_20260602/plots/cylindrical_symmetry_vs_rigidity.png`

and:

`/Users/yueying/Workspace/SLIMED/analysis/cluster_results_20260602/plots/cylindrical_symmetry_vs_rigidity.svg`

The x-axis is protein lattice rigidity on a log scale. The y-axis is the dimensionless cylindrical-asymmetry metric. Marker shape and color distinguish eccentricity:

- black circles: `e = 0`
- gray squares: `e = 0.866`

Error bars are one standard error from the joint minimum-energy bootstrap described above.
