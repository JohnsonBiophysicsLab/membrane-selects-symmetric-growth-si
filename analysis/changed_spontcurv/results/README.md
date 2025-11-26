# Dataset README

## Dataset Information
- **Time of Dataset Generation:** 12:17 AM Thu Nov 20, 2025
- **Author:** yying7@jh.edu
- **Purpose of Dataset:**
  - This dataset is generated with the constant **shape and area** of `COM.csv` (corresponding to protein scaffolding), while only varying the **density of nodes** to analyze the **bending energy** of the scaffolded membrane under **different spontaneous curvature**.
- **Note on SVG files**
  - Most of SVG files are not included but can be generated from `.ipynb` scripts. This is due to SVG files taking too much space.
- **Supercomputer Cluster Used:** ARCH Rockfish at Johns Hopkins University

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
     - **Volume constraint energy:**
       \[ E_V = \frac{1}{2} \mu_V \frac{(V-V_0)^2}{V_0} \]
     
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


