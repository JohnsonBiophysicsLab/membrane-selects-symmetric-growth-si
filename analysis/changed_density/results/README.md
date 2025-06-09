# Dataset README

## Dataset Information
- **Time of Dataset Generation:** 1740671962 Seconds since Jan 01 1970. (UTC) 10:59 AM Thu Feb 27
- **Author:** yying7@jh.edu
- **Purpose of Dataset:**
  - This dataset is generated with the constant **shape and area** of `COM.csv` (corresponding to protein scaffolding), while only varying the **density of nodes** to analyze the **bending energy** of the scaffolded membrane under **different node densities**.
  - This dataset is to correct the previous Feb 25 one, where I reverted a few changes in the energy force calculation and we need to investigate into that later. (https://github.com/mjohn218/continuum_membrane/commit/285b1dedd5577431386844f97083a33272f861cf)
- **Note on SVG files**
  - Most of SVG files are not included but can be generated from `.ipynb` scripts. This is due to SVG files taking too much space.
- **Supercomputer Cluster Used:** ARCH Rockfish at Johns Hopkins University

## Geometry Information

a=30 nm; b=30 nm (circular)

:wc -l mc_mesh_a30_b30d*

<-- GagSize = number of rows in file

 GagSize FileName               TrialGroup
     162 mc_mesh_a30_b30d10.csv 1
     112 mc_mesh_a30_b30d12.csv 2
      72 mc_mesh_a30_b30d15.csv 3
      45 mc_mesh_a30_b30d20.csv 4
      30 mc_mesh_a30_b30d25.csv 5
     241 mc_mesh_a30_b30d8.csv  0
     977 total

d10 means the mean distance between nearby gags is 1.0 * 4.7
d25 means the mean distance between nearby gags is 2.5 * 4.7
etc.

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

3. **Membrane Brownian Dynamics Simulation:**
   - The **moving membrane surface** is simulated using a **displacement equation**:
     \[ \Delta X = -\frac{D\Delta t}{k_b T} \nabla E + \sqrt{2D\Delta t} (N(0,1)) \]
   - The displacement occurs on the **limit surface**, not the control mesh.

4. **Boundary Conditions:**
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


