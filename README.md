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



## `src/baseline/` — Backprop Reference Implementation

Standard supervised learning pipeline using cross-entropy loss and a single optimizer over the whole network. This is the baseline that Forward-Forward is compared against.

### File reference

| File | Description |
|------|-------------|
| **`model.py`** | Defines `DenseNetConfig` (dataclass) and `DenseNet` (MLP). Supports configurable depth, width, activation (`relu`/`tanh`/`gelu`), dropout, and number of classes. Includes forward hooks for capturing activations/gradients during debugging. No output softmax — returns raw logits. |
| **`trainer.py`** | Contains `TrainerConfig` (epochs, lr, optimizer type, scheduler, early stopping patience), `_run_epoch()` (single train/eval pass), and the `Trainer` class that wraps the full training loop with validation, checkpointing best weights, and test-set evaluation. Supports Adam or SGD optimizers. |
| **`main.py`** | Entry point for running baseline training. Loads data via `utils.data_loader`, instantiates a `DenseNet` model, sets up optimizer + loss, creates a `TrainerConfig`, then trains, evaluates on the test set, and prints a summary. |



To change **model depth/width**, edit `DenseNetConfig` in `src/baseline/model.py`:
```python
hidden_dims: List[int] = field(default_factory=lambda: [500, 500, 500])
# e.g. for a deeper model: [512, 512, 512, 512, 512]



---

## `src/forward/` — Forward-Forward Implementation

Implements Hinton's Forward-Forward algorithm where each layer learns independently using a local goodness loss. Instead of backprop through the whole network, positive (correct label) and negative (wrong label) samples are fed forward, and each layer adjusts its weights to increase goodness for positives and decrease it for negatives.



### File reference

| File | Description |
|------|-------------|
| **`model.py`** | Defines `ForwardNetConfig` and `ForwardNet`. Each hidden layer L2-normalizes its input before the linear map, applies an activation (`relu`/`tanh`/`gelu`), then computes a per-layer goodness = mean of squared post-activation units. The local loss is a logistic function around threshold θ=2.0 (Hinton's value). `ff_forward()` returns one loss tensor per layer for training; `forward()` returns per-layer goodnesses for inference. No output head — classification is done by summing goodness across layers with all possible label stampings. |
| **`data_loader.py`** | Defines `FFDataConfig` (extends `DataConfig`), `FFDataset` (wraps any image dataset and emits `(positive, negative, label)` triples via one-hot label stamping), and `FFInferenceDataset` (stamps every possible label per sample so the network can pick the highest-goodness one). Supports `"wrong_label"` and `"hybrid"` negative strategies. |
| **`trainer.py`** | Two core functions: `ff_epoch()` runs one training epoch — concatenates pos+neg samples, computes all layer losses via `model.ff_forward()`, then does per-layer backward + step; `ff_accuracy()` evaluates on inference data by computing goodness for each label-candidate and picking the argmax. |
| **`main.py`** | Entry point for FF training. Builds data loaders with `FFDataset`/`FFInferenceDataset`, instantiates a `ForwardNet` with configurable depth/activation, sets up per-layer Adam optimizers (lr=0.03), then trains epoch-by-epoch printing per-layer losses and validation accuracy, evaluates on test set, and saves checkpoints. |