# membrane-selects-symmetric-growth-si

This repository contains the supporting source code for graphing, analytical, and numerical calculations related to the continuum membrane model by Dr. Margaret E Johnson lab, JHU.

## Dependencies
Most scripts (`.ipynb`) are written in python3 and the python environment is described below. Note that some of the scripts with `.nb` extension are Mathematica scripts and those scripts are written and run with Wolfram `Mathematica 14.0.0.0` on `Mac OS X ARM (64-bit)`.

 The following dependencies are required to run the python codes in this repository. The specified versions are provided for reference and are not strictly required; however, if issues arise, please ensure that your dependencies resolve to the same versions.

- Python 3.9.7
- PyVista 0.38.3
- Matplotlib 3.7.0
- Pandas 1.3.4
- Scipy 1.13.1
- Seaborn 0.12.1

To set up the environment, use the provided `environment.yml` file:

```sh
conda env create -f environment.yml
conda activate my_env
```

## Data

There are three datasets managed by Github LFS under [`data/`](./data).

- [`changed_density_result.zip`](./data/changed_density_result.zip) : dataset for membrane conformation at minimum total energy when attached to protein lattices of different protein densities with the same perimeter.
- [`changed_ecc_result.zip`](./data/changed_ecc_result.zip) : dataset for membrane conformation at minimum total energy when attached to protein lattices of different protein eccentricities and sizes with the same density.
- [`nerdss_mesh_result.zip`](./data/nerdss_mesh_result.zip) : dataset for membrane conformation at minimum total energy when attached to gag lattices stochastically assembled by the NERDSS software.

## Analysis

[`analysis/`](./analysis) contains scripts for numerical and analytical analysis as well as plotting scripts.

- [`general`](./analysis/general) includes all scripts for short analytical and numerical analysis and scripts that draw graphical representation of the system as well as the simulation scheme. 
- [`changed_density`](./analysis/changed_density), [`changed_ecc`](./analysis/changed_ecc), and [`nerdss_mesh`](./analysis/nerdss_mesh) correspond to the three datasets in [`data/`](./data). See the section above for more information.

## SVG

[`svg/`](./svg) has vector graphics version of the plots in the main paper as well as in the SI.

## meshgen

[`meshgen/`](./meshgen) contains a script for generating ideal protein lattice cap with controlled density and perimeter via Lennard-Jones Potential based Monte-Carlo method. The results are store under [`output`](./meshgen/output) as `.csv` files.

## Related Software
### NERDSS

NERDSS (Nonequilibrium Simulator for Multibody Self-Assembly at the Cellular Scale) is a computational tool designed for simulating self-assembly dynamics of biomolecular systems in nonequilibrium conditions.

For more information, visit the official repository: [NERDSS GitHub Page](https://github.com/mjohn218/NERDSS)

### Citation:

#### This work:

Ying, Y.M. & Johnson, M.E. Membrane bending energy selects for compact growth of protein assemblies. (in preparation)

#### Models:

**Continuum Membrane**: Fu, Y., Zeno, W.F., Stachowiak, J.C., Johnson, M.E. (2021). A continuum membrane model can predict curvature sensing by helix insertion. *Soft Matter*, 2021, 17, 10649 - 10663. DOI: https://doi.org/10.1039/D1SM01333E.

**NERDSS**: Varga, M. J., Fu, Y., Loggia, S., Yogurtcu, O. N., & Johnson, M.E. (2020). NERDSS: A Nonequilibrium Simulator for Multibody Self-Assembly at the Cellular Scale. *Biophys. J.*, 118(12), 3026-3040. https://doi.org/10.1016/j.bpj.2020.05.014

**Stochastically assembled Gag lattice**: Qian, Y., Evans, D., Mishra, B., Fu, Y., Liu, Z.H., Guo, S., Johnson, M.E. Temporal control by cofactors prevents kinetic trapping in retroviral Gag lattice assembly. *Biophys. J.* 2023 Aug 8;122(15):3173-3190. doi: 10.1016/j.bpj.2023.06.021. Epub 2023 Jun 30. PMID: 37393432; PMCID: PMC10432227.

## Continuum Membrane Model
The continuum membrane model is currently available at the following repository:
[Continuum Membrane Model GitHub](https://github.com/Yibenfu/continuum_membrane_model)

Further details on this model will be added as development progresses.
