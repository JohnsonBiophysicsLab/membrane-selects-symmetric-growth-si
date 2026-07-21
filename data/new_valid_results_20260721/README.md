# New Valid Results, July 2026

This folder contains only the newest valid result artifacts needed for the
July 2026 supplemental figure updates. It intentionally does not mirror the
full local/cluster result folders.

Included:

- `rigidity_energy_curvature/`: compact CSV summaries for the updated
  a30_b30/a30_b15 rigidity comparison. The a30_b15 rows include the fixed
  K=1000 rerun (`s1000_from_s316_anneal_20260616`) and exclude overwritten or
  failed K=1000 attempts.
- `curvature_asymmetry/`: compact candidate, selected-minimum, outlier-flag,
  and final-summary CSVs for cap curvature-difference and cylindrical
  asymmetry analyses.
- `selected_minimum_surfaces/`: compressed arrays for only the 10
  minimum-energy selected surfaces used in the mean-curvature heatmap panel.
  These are not full trajectory folders.
- `hoomd_hiv_gag_geometry/`: HOOMD-derived canonical surface summaries,
  second-half mean surface arrays, per-frame feature CSVs, and GSD-derived Gag
  COM positions used for the HIV Gag geometry/curvature panel.
- `MANIFEST.csv`: file-level provenance and notes.

Excluded:

- Full `cluster_results_20260602/fetched/` folders.
- Full `~/Downloads/change_stiffness` folders.
- Raw HOOMD `.gsd` trajectories.
- Failed, incomplete, superseded, or overwritten restart attempts.
- Previous broad zip archives containing unrelated old and new results.

Some compact derived tables contain both `e = 0` and `e = 0.866` rows because
the uploaded plots compare those two conditions. The full old raw sweep data
are not recopied here.
