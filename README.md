# Forward-Forward

An implementation of Hinton's [Forward-Forward](https://arxiv.org/abs/2212.13345) algorithm, benchmarked against a standard backprop baseline on MNIST. Both implementations use identical data pipelines and matched architectures for fair comparison.

## Quick Start

### Install `uv` (package manager)

If you don't already have [uv](https://docs.astral.sh/uv/) installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or with Homebrew
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Sync dependencies

```bash
uv sync
```

This creates a virtual environment and installs all required packages (`torch`, `torchvision`, `matplotlib`, `torchviz`, etc.) as declared in [`pyproject.toml`](pyproject.toml).

### Install custom scripts (editable)

To make your custom Python modules importable throughout the project:

1. **Add `__init__.py`** to each folder you want to treat as a package:
   ```bash
   touch src/your_module/__init__.py
   ```

2. **Editable install** so imports resolve in-place (no copy):
   ```bash
   uv pip install -e .
   ```