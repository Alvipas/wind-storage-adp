# Prompt: Research Paper Study Repo

Use this prompt to implement a research paper from a PDF. Replace `<PAPER_PATH>` and `<REPO_NAME>` before starting.

---

## Task

I want to study and implement the paper at `<PAPER_PATH>`.

Create a structured repository called `<REPO_NAME>` with:
1. Rigorous markdown documentation of the paper's formulation
2. Python implementation combining reusable modules and Jupyter notebooks
3. A reproducible `uv` environment for running notebooks
4. A GitHub repository pushed to my account

---

## Phase 0 — Environment setup with `uv`

Before writing any code, set up the Python environment using [`uv`](https://docs.astral.sh/uv/).

### Create the project

```bash
mkdir <REPO_NAME> && cd <REPO_NAME>
uv init --no-package
```

This creates `pyproject.toml` and `.python-version`. Edit `pyproject.toml` to set the project metadata and core dependencies:

```toml
[project]
name = "<REPO_NAME>"
version = "0.1.0"
description = "<one-line description>"
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "jupyterlab>=4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "ruff>=0.1",
    "ipython>=8.0",
]
```

Add further dependencies as needed with `uv add <package>`.

### Sync the environment and generate the lock file

```bash
uv sync
```

This creates `.venv/` and `uv.lock`. The lock file pins exact versions for reproducibility — commit it.

### Run notebooks

```bash
# Option A — without activating the venv (recommended)
uv run jupyter lab notebooks/

# Option B — activate first, then run normally
source .venv/bin/activate   # Windows: .venv\Scripts\activate
jupyter lab notebooks/
deactivate
```

### Create `ENVIRONMENT.md`

Write a short `ENVIRONMENT.md` in the repo root covering:
- How to install `uv` (curl installer + brew/apt/pacman alternatives)
- `uv sync` to create the environment
- `uv run jupyter lab notebooks/` to launch notebooks
- `uv sync --frozen` for read-only installs (CI)
- Note: `.venv/` is gitignored; `uv.lock` is committed

### `.gitignore` additions for `uv`

```
.venv/
.python-version   # optional: commit if you want to pin the Python version
```

---

## Phase 1 — Read the paper

Read the PDF in chunks (max 20 pages per call, use the `pages` parameter). Cover:
- Abstract and introduction: problem setting, motivation, contributions
- Model formulation: variables, sets, constraints, objective
- Solution method(s): algorithm(s), key equations
- Numerical setup: parameters, baselines, experimental design
- Results: tables and figures to replicate

---

## Phase 2 — Plan (confirm before implementing)

Propose a directory layout:

```
<REPO_NAME>/
├── docs/              # one .md file per major paper section
├── src/               # pure Python modules, no framework dependencies
├── notebooks/         # one .ipynb per module/topic
├── prompts/           # this prompt file, for reuse
├── pyproject.toml     # uv project definition and dependencies
├── uv.lock            # pinned dependency versions (committed)
├── ENVIRONMENT.md     # how to set up and run the environment
├── README.md
└── .gitignore
```

For each file, state:
- **What** it covers (equations/sections from the paper)
- **Dependencies** on other src files (determines write order)

Wait for approval before writing any code.

---

## Phase 3 — Documentation (`docs/`)

Write one markdown file per major section. Rules:
- Use **exact notation** from the paper (same symbols, equation numbers, table numbers)
- Be **concise**: state definitions, equations, and their interpretation — no paraphrasing
- Include a notation summary table at the end of each file
- Reference equation numbers as `(eq. N)` matching the paper

Suggested sections (adjust to the paper):
- `01_problem_formulation.md` — sets, variables, constraints, objective
- `02_stochastic_model.md` — uncertainty model, distributions, processes
- `03_solution_method.md` — algorithm(s), pseudocode, convergence
- `04_experiments.md` — parameters, baselines, metrics, key results

---

## Phase 4 — Python modules (`src/`)

Write modules in **dependency order** (no module imports one written after it).

Guidelines:
- Each module is a single coherent responsibility (data model, dynamics, solver, algorithm)
- Use dataclasses for parameter bundles
- Functions are pure where possible; side effects only in top-level runners
- No comments explaining *what* the code does — only *why* if non-obvious
- Match variable names to paper notation (e.g. `theta_Y`, `mu_eps_P`)
- Validate constraints at construction time (raise `ValueError` with a clear message)

Typical module order:
1. `params.py` / `model.py` — dataclasses for all parameters
2. `stochastic.py` — random process simulation
3. `environment.py` / `dynamics.py` — state transitions and rewards
4. `discretisation.py` — grid construction and probability tables (if needed)
5. `benchmark.py` — exact / LP / baseline solution
6. `algorithm.py` — main solution method (ADP, RL, stochastic programming, etc.)

---

## Phase 5 — Jupyter notebooks (`notebooks/`)

Write one notebook per topic. Rules:
- First cell: `sys.path.insert(0, '../src')` then imports
- Keep cells short; one concept per cell
- Every notebook must be **independently runnable** top-to-bottom
- Each notebook has a markdown header stating goals and which paper figures/tables it replicates
- Use reduced hyperparameters (smaller N, fewer iterations) for quick demo; add a note to increase for full replication

Suggested notebooks (adjust to the paper):
- `01_stochastic_model.ipynb` — simulate paths, verify moments, sensitivity plots
- `02_dynamics.ipynb` — state transitions, reward surface, edge cases
- `03_benchmark.ipynb` — exact/LP solution, policy visualization
- `04_main_algorithm.ipynb` — run algorithm, convergence, policy plots, reward comparison
- `05_sensitivity.ipynb` — parameter sweep, metamodel, replicate sensitivity tables/figures

---

## Phase 6 — Repo setup

```bash
git init
git branch -m master main

# Stage everything except excluded files
git add docs/ src/ notebooks/ prompts/
git add pyproject.toml uv.lock ENVIRONMENT.md README.md .gitignore

git commit -m "Initial implementation of <Author> (<Year>)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

gh repo create <REPO_NAME> --public \
  --description "<one-line description>" \
  --source=. --remote=origin --push
```

Do **not** commit: the source PDF, `.claude/`, `.venv/`, data files (`*.pkl`, `*.npy`, `*.csv`), or output directories.

---

## Quality checklist

Before pushing, verify:
- [ ] `uv sync` completes without errors on a clean checkout
- [ ] `uv run jupyter lab notebooks/` launches without errors
- [ ] All equations in docs match the paper (spot-check 3–5 key ones)
- [ ] Each src module is importable without error (`uv run python -c "import src.<module>"`)
- [ ] Each notebook runs top-to-bottom without error (with reduced hyperparameters)
- [ ] `uv.lock` is committed; `.venv/` is gitignored
- [ ] `ENVIRONMENT.md` covers install → sync → run in three steps
- [ ] README correctly describes the directory structure and links to `ENVIRONMENT.md`
- [ ] `.gitignore` covers Python, Jupyter, uv, and IDE artifacts
- [ ] No hardcoded absolute paths in any file
