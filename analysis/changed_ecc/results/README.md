# Dataset README

## Dataset Information
- **Time of Dataset Generation:** (UTC) 2:30 PM Tue Mar 11
- **Author:** yying7@jh.edu
- **Purpose of Dataset:**
  - This dataset is generated with varied **shape and area** of `COM.csv` (corresponding to protein scaffolding)analyze the **bending energy** of the scaffolded membrane under different gag lattice **size** (defined by major axis length) and **perimeter eccentricity**.
  - This dataset is to correct the previous Feb 25 one, where I reverted a few changes in the energy force calculation and we need to investigate into that later. (https://github.com/mjohn218/continuum_membrane/commit/285b1dedd5577431386844f97083a33272f861cf)
- **Parameter space**
  - 5 different eccentricities paired with 6 different sizes.
- **Supercomputer Cluster Used:** ARCH Rockfish at Johns Hopkins University

## Geometry Information

### Input file naming

Note, the input file are named after the major and minor axis length in nanometers
of the gag lattice, i.e. the `a` and `b` in the equation

`x^2 / a^2 + y^2 / b^2 = 1`

e.g. `...a40_b30.csv` refers to the gag lattice defined in the oval shape:

x^2 / 40^2 + y^2 / 30^2 = 1

### Eccentricities

The eccentricity are defined as ecc = sqrt(1 - minor_axis^2 / major_axis^2) given in `TrialGroup`

**The examples given below are with major axis = 40 nm**
TrialGroup 0: ecc = 0.0 (40 nm,40 nm)
TrialGroup 1: ecc = 0.66143782776 (40 nm,30 nm)
TrialGroup 2: ecc = 0.8 (40 nm,24 nm)
TrialGroup 3: ecc = 0.86602540378 (40 nm,20 nm)
TrialGroup 4: ecc = 0.91651513899 (40 nm,16 nm)

The groups are divided into trial and size group. Trial groups are based on different eccentricity
shown above. Size groups are defined by the length of oval major axis (in this case `a`).

### Sizes

The size group is given in in terms of major axis length.

SizeGroup 10: major axis length = 10.0 nm
SizeGroup 20: major axis length = 20.0 nm
SizeGroup 25: major axis length = 25.0 nm
SizeGroup 30: major axis length = 30.0 nm
SizeGroup 35: major axis length = 35.0 nm
SizeGroup 40: major axis length = 40.0 nm

## Note for extra trials for size group 40 nm

Extra trial for size group 40 nm

Note that I ran extra trial for the major axis = 40 nm group, because the previously used range of S0
for all other groups causes simulation in this group to not converge. Thus another trial `ecca41` was run,
still with a = 40 nm (just changed the name to 41 to avoid naming conflict issue in auto-pipeline from
supercomupter cluster). Therefore, there is a part of this script to concat `ecca41` back into `ecca40`.
Hence, the sample number of `ecca40` is 16 now, compared to 8 for trials in all other size groups.


## Model Summary
This dataset is generated using the **NERDSS - Continuum Membrane & Dynamics tool**, which models **continuum membranes** using a triangular mesh and optimizes their structure through an **energy function**. The key steps of the model include:

1. **Triangular Mesh Construction:**
   - The membrane is approximated using a **triangular mesh**.
   - The mesh is refined using **Loop's subdivision method**.
   
2. **Energy Minimization:**
   - The membrane energy is minimized using an **energy function**, which includes:
     - **Bending energy:**
       \[ E_B = \int_S \frac{1}{2}\kappa (2H-C_0)^2 dS \]
     - **Area constraint energy:**
       \[ E_S = \frac{1}{2} \mu_S \frac{(S-S_0)^2}{S_0} \]
     
   where \( \kappa \) is the membrane bending constant, \( H \) is the mean curvature, and \( C_0 \) is the spontaneous curvature.

3. **Boundary Conditions:**
   - **Periodic:** Three rings of ghost vertices mimic opposite side movement.

## Related GitHub Issue (Reason for Rerun)
This dataset was regenerated due to a **substantial bug** in the **OMP parallelization of energy summation**.
- The bug caused **energy minimization results to vary by 2x - 5x** on **Rockfish vs. local machines/workstations**.
- The issue was traced to **undefined behavior in OpenMP parallelization**, leading to **incorrect synchronization across cores on Rockfish**.
- The bug depended on the **number of nodes and cores used**, affecting correctness.
- A **new version** of the model was pushed to GitHub, and all affected jobs were **rerun** with the corrected implementation.

## Usage and Access
- This dataset is meant for **analyzing bending energy** across different node densities.
- For any issues or reproducibility concerns, contact **yying7@jh.edu**.

## License and Citation
- This dataset follows the **GNU Public License**.
- If using or modifying the **continuum membrane model**, please cite:
  - **Fu et al., J Chem Phys (2019)**: "An implicit lipid model for efficient reaction-diffusion simulations of protein binding to surfaces of arbitrary topology."
  - **Fu et al., bioRxiv (2021)**: "A continuum membrane model predicts curvature sensing by helix insertion."

## References
- **Helfrich, W. (1973):** "Elastic properties of lipid bilayers: theory and possible experiments." *Zeitschrift für Naturforschung C, 28(11), 693-703.*


