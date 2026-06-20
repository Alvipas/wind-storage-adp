# wind-storage-adp

Python implementation and documentation of:

> Löhndorf, N. & Minner, S. (2009). *Optimal Day-Ahead Trading and Storage of Renewable Energies — An Approximate Dynamic Programming Approach.*

The paper models the optimal bidding strategy for a hybrid wind power + energy storage system as a continuous-state infinite-horizon MDP, solved via Least Squares Approximate Policy Iteration (LSAPI).

---

## Structure

```
docs/          Rigorous markdown summaries of each paper section (notation-faithful)
src/           Python modules (pure functions, no framework dependencies)
notebooks/     Jupyter notebooks demonstrating each component
```

### Documentation

| File | Content |
|------|---------|
| [docs/01_mdp_formulation.md](docs/01_mdp_formulation.md) | State space, storage dynamics, reward function, Bellman equation |
| [docs/02_stochastic_processes.md](docs/02_stochastic_processes.md) | AR(1) supply & price processes, cross-correlation |
| [docs/03_solution_methods.md](docs/03_solution_methods.md) | Discrete transition matrix, LP benchmark, LSPE, LSAPI algorithm |
| [docs/04_value_function_approx.md](docs/04_value_function_approx.md) | Polynomial basis functions, default parameters, results |

### Source modules

| Module | Responsibility |
|--------|---------------|
| [src/stochastic.py](src/stochastic.py) | AR(1) simulation, parameter derivation (eqs. 6–12) |
| [src/environment.py](src/environment.py) | Storage dynamics, reward function (eqs. 2–5) |
| [src/transition_matrix.py](src/transition_matrix.py) | Discrete conditional probabilities, joint transition matrix (eqs. 13–18) |
| [src/basis_functions.py](src/basis_functions.py) | 2nd- and 3rd-order polynomial basis functions (eqs. 26–27) |
| [src/lp_benchmark.py](src/lp_benchmark.py) | LP formulation and solver (eqs. 19–20) |
| [src/lsapi.py](src/lsapi.py) | LSAPI algorithm (Figure 1, eqs. 21–25) |

### Notebooks

| Notebook | Demonstrates |
|----------|-------------|
| [01_processes.ipynb](notebooks/01_processes.ipynb) | AR(1) paths, moment verification, sensitivity to $\theta_Y$, $\rho$ |
| [02_environment.ipynb](notebooks/02_environment.ipynb) | Storage dynamics, reward surface, optimal myopic bid |
| [03_lp_benchmark.ipynb](notebooks/03_lp_benchmark.ipynb) | Transition matrix, LP solution, policy contour plots (Figs. 2a, 3a) |
| [04_lsapi.ipynb](notebooks/04_lsapi.ipynb) | LSAPI-2 and LSAPI-3, policy plots (Figs. 2b–c, 3b–c), Table 2 |
| [05_sensitivity.ipynb](notebooks/05_sensitivity.ipynb) | Space-filling design, metamodels $\hat{Q}_1$/$\hat{Q}_2$, Tables 4–5 |

---

## Dependencies

```
numpy
scipy
matplotlib
pandas
scikit-learn
jupyter
```

Install with:
```bash
pip install numpy scipy matplotlib pandas scikit-learn jupyter
```

---

## Quick start

```bash
cd notebooks
jupyter notebook 01_processes.ipynb
```

To replicate the paper's results, increase the LSAPI hyperparameters to `N=500, M=200, buffer_size=25000` in notebooks 04 and 05.
