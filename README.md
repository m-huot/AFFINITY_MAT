# Optimal Protocols for Vaccination

This repository contains tools to optimize vaccination schedules.

## Methodology

The framework relies on three main components to find the optimal antigen concentration schedule $C^*(t)$:

* **Fokker-Planck Simulations:** Models the probability density of B-cell affinity evolution over time.
* **Stochastic Lineage Tracking:** Captures the evolutionary history and stochastic fluctuations of B-cell populations within Germinal Centers.
* **Least-Action Optimization:** Identifies the most probable evolutionary pathways leading to high-affinity states and optimizes antigen administration profiles to favor these paths.

## Key Components

* `fokker_planck.py`: Implements continuity equation.
* `stochastic_simul.py`: Implements ancestry-aware simulations to track lineage maturation.
* `least_action.py`: Provides routines to solve for the optimal trajectory and compute the global Hessian.
* `script_vaccine_optim.py`: The script for computing optimized immunization parameters based on target affinity and decay time ($\tau$).

Please consider citing:

Huot, M., Molari, M., Monasson, R., & Cocco, S. (2025).  **Optimal Maturation Protocols for High-Affinity Antibody Targets: A Path-Integral Approach** .  *bioRxiv* , 2025.12.21.695799. [https://doi.org/10.64898/2025.12.21.695799](https://www.google.com/search?q=https://doi.org/10.64898/2025.12.21.695799)
