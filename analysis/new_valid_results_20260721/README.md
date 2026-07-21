# New Valid Results Analysis, July 2026

This folder contains the analysis scripts and generated plots for the newest
valid result set uploaded under `data/new_valid_results_20260721/`.

Primary entry point:

```bash
python plot_uploaded_results.py
```

The script reads only repository-relative paths and writes regenerated figures
to `analysis/new_valid_results_20260721/recomputed_outputs/`.

The `outputs/` folder contains the plot files generated during the local
analysis session. The `source_scripts/` folder preserves the source scripts
used during analysis; some of those preserve original workstation paths for
provenance, while `plot_uploaded_results.py` is the portable repository-local
replotting script.

The upload is deliberately small: it includes compact result tables, selected
surface arrays, and HOOMD-derived canonical arrays, not full simulation
trajectories or broad old/new archives.
