# Environment Setup

## Using `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package installer and resolver written in Rust.

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via your package manager:
```bash
# macOS
brew install uv

# Linux (Ubuntu/Debian)
sudo apt install uv

# Arch
sudo pacman -S uv
```

### Create and activate the environment

```bash
# Create the venv and install dependencies (one-liner)
uv sync

# Activate the venv
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### Deactivate the environment

```bash
deactivate
```

---

## Using standard `pip`

If you prefer traditional venv + pip:

```bash
# Create venv
python3.10 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install -e '.[dev]'

# (on Windows)
# .venv\Scripts\activate
```

---

## Running Jupyter notebooks

Once the environment is activated:

```bash
jupyter lab notebooks/
```

Or use VS Code with the Python extension pointing to `.venv/bin/python`.

---

## Checking the lock file

The `uv.lock` file is generated from `pyproject.toml` and ensures reproducible installs across machines.

Regenerate it after modifying `pyproject.toml`:

```bash
uv sync
```

For read-only installs (CI, production):

```bash
uv sync --frozen
```
