# Monte Carlo Mesh Generator for Elliptical Spherical Caps

Author: Yue Moon Ying, Dr. Samuel Foley

This repository provides a Monte Carlo-based simulation to generate evenly spaced points on a spherical cap shaped by an elliptical boundary. The method uses repulsive particle dynamics and adaptive point addition/removal to approximate uniform surface coverage, intended for modeling protein lattice or mesh distributions on curved biological membranes.

## Directory Structure

```

.
├── mc\_meshgen.py        # Main Python script to generate meshes and animations
├── output/              # Contains CSV files of final 3D point coordinates
│   ├── mc\_mesh\_a30\_b20.csv
│   ├── mc\_mesh\_a30\_b30d10.csv
│   └── ...
├── anim/                # Contains animated GIFs of the simulation process
│   ├── moc\_mesh\_a30\_b20d10.gif
│   └── ...

```

## File Naming Convention

The output files (in both `output/` and `anim/`) follow the naming scheme:

```

mc\_mesh\_a{MAJOR}\_b{MINOR}d{DENSITY}.csv  # for point data
monte\_carlo\_mesh\_a{MAJOR}\_b{MINOR}d{DISTANCE}.gif  # for animations

````

Where:
- `MAJOR` is the major axis of the elliptical cap (in nanometers)
- `MINOR` is the minor axis of the elliptical cap (in nanometers)
- `DISTANCE` is a scaling factor for protein-protein distance 
  - `d10` means default density  (set to 4.7 nm)
  - `d20` means 2× protein-protein distance 
  - `d5` means half the protein-protein distance

## How to Run

To execute the simulation, modify the parameters in `mc_meshgen.py` under the `if __name__ == "__main__"` block. Example parameters:
```python
sim = ParticlePlacementSimulation(
    sphere_radius=50.0,
    ellipse_major_axis=30.0,
    ellipse_minor_axis=20.0,
    desired_area_per_point=19.13 * scale * scale,
    scale=1.0,
    max_iteration=2000,
    dt=1.0
)
sim.run_simulation()
````

This will:

* Save the converged 3D coordinates to a `.csv` file in `output/`
* Generate a `.gif` of the simulation in `anim/`

## Features

* Adaptive node count control based on local spacing
* Lennard-Jones-like repulsive potential
* Monte Carlo perturbation with probabilistic acceptance
* Optional animation of simulation trajectory
* Automatic output folder creation

## Acknowledgements

Special thanks to **Dr. Sam Foley** (http://samuelfoley.com/) for developing the original prototype of this simulation tool and for valuable guidance on its conceptual and computational foundations.

## Dependencies

* Python 3.x
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`
* `celluloid`

You can use the `.yml` file for configuring a conda environment. Alternatively, if you want to just run the `mc_meshgen.py`, install via pip if needed:

```bash
pip install numpy scipy pandas matplotlib celluloid
```
