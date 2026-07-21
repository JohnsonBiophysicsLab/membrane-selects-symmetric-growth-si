# Ellipsoid-Sphere Deviation Metric

This analysis asks whether the membrane cap is well described by the intended
50 nm radius spherical geometry after protein-lattice relaxation.

For each trajectory, the script:

1. Loads the membrane control mesh snapshot nearest the minimum logged total
   energy (`Min_Etot`).
2. Centers the mesh in x/y and shifts the lowest membrane point to `z = 0`,
   matching the side-view and cylindrical-asymmetry analyses.
3. Approximates the Loop subdivision surface with two subdivision iterations.
   This is sufficient for a global ellipsoid fit while avoiding the runtime
   cost of fitting every triangle from a third refinement pass.
4. Keeps only the cap region with projected radius `r <= 30 nm`, after also
   excluding square-box boundary regions that cannot support a complete
   circular cap.
5. Fits an axis-aligned ellipsoid to a deterministic, evenly distributed subset
   of the cap-surface triangle centroids using area-weighted nonlinear least
   squares.

The fitted ellipsoid has semi-axes `(a_x, a_y, a_z)`.  Because only the cap is
available, the full semi-axis lengths are not uniquely constrained: many large
ellipsoids can share nearly the same cap curvature.  The physically stable
quantities are therefore the fitted ellipsoid's apex curvature radii:

```text
R_x = a_x^2 / a_z
R_y = a_y^2 / a_z
```

The primary scalar metric is the RMS fractional deviation of these two fitted
curvature radii from an ideal 50 nm sphere:

```text
deviation = 100 * sqrt(mean(((R_i - 50 nm) / 50 nm)^2))
```

The reported value is therefore a percent.  It is zero for a cap whose fitted
ellipsoid has the same local curvature radii as a 50 nm sphere.  Larger values
mean the cap is locally too flat, too curved, or anisotropic relative to the
target sphere.

The script also reports two size-independent shape metrics based on the same
fitted apex radii:

```text
closest_sphere_radius = mean(R_x, R_y)
closest_sphere_deviation = 100 * sqrt(mean(((R_i - closest_sphere_radius)
                                            / closest_sphere_radius)^2))
ellipsoid_eccentricity = sqrt(1 - min(R_x, R_y)^2 / max(R_x, R_y)^2)
```

These ignore whether the cap is larger or smaller than 50 nm and instead ask
whether the fitted ellipsoid cap is circular/spherical after choosing its own
best radius.

The fit is kept axis-aligned because the simulation coordinate frame is
physically meaningful: x/y are the lattice axes and z is the membrane-height
axis.  Allowing arbitrary rotations made the cap-only fit less interpretable
without changing the spherical reference, which has no preferred orientation.

Uncertainty is computed with the same joint row-wise bootstrap used in the
rigidity and curvature analyses.  Within each `(eccentricity, rigidity)` group,
whole trajectories are resampled with replacement; the minimum-`Etot` row in
each bootstrap sample is selected; and the ellipsoid metric from that same row
is recorded.  The plotted error bar is one bootstrap standard error.
