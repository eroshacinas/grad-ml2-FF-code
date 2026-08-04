from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F



# CONFIG
@dataclass
class ForwardNetConfig:
    input_dim:    int         = 784 # 28×28 flattened MNIST
    hidden_dims:  List[int]   = field(default_factory=lambda: [500, 500, 500, 500])  # baseline dims
    num_classes:  int         = 10
    activation:   str         = "relu" # relu, tanh, gelu
    dropout:      float       = 0.0
    theta:        float       = 2.0 # based on hinton's value
    norm_eps:     float       = 1e-4


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class ForwardNet(nn.Module):
    """
    fully-connected Forward-Forward network
      - each layer L2-length-normalizes its INPUT before the linear map
      - goodness = MEAN of squared post-activation units
      - per-layer logistic loss around a fixed threshold θ = 2.0 (based on hinton)
    """

    def __init__(self, cfg: ForwardNetConfig = ForwardNetConfig()):
        super().__init__()
        self.cfg = cfg

        if cfg.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{cfg.activation}'. Choose from {list(_ACTIVATIONS)}"
            )

        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0.0 else None

        # variable-depth hidden layers
        dims = [cfg.input_dim] + cfg.hidden_dims
        self.hidden_layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(cfg.hidden_dims))
        ])

        # one activation module per layer -- matches baseline (each Layer has its
        # own ReLU) and lets hooks capture post-activation output per layer
        self.activations = nn.ModuleList([
            _ACTIVATIONS[cfg.activation]() for _ in cfg.hidden_dims
        ])

        # NOTE: no output/softmax head. FF classifies by accumulated goodness.

        # fixed threshold, moves with the model, saved/loaded, never optimized
        self.register_buffer("theta", torch.tensor(cfg.theta))

        # hook bookkeeping
        self._hooks: list = []
        self._activations: dict[str, torch.Tensor] = {}
        self._gradients: dict[str, torch.Tensor] = {}

    #  normalization + goodness
    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """project each sample onto its unit L2 direction (baseline: x / ‖x‖^2)"""
        return x / (x.norm(dim=1, keepdim=True) + self.cfg.norm_eps)

    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return (x ** 2).mean(dim=1)


    # inference forward: return per-layer goodness 
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        full forward pass for evaluation / goodness-based prediction
        returns one goodness tensor (B,) per hidden layer
        """
        goodnesses = []
        h = x
        for linear, act in zip(self.hidden_layers, self.activations):
            h = self._l2_normalize(h) # normalize 
            z = act(linear(h)) # post-activation
            goodnesses.append(self.goodness(z))
            h = z # pass raw output; next layer normalizes it

            if self.dropout is not None:
                h = self.dropout(h)
        return goodnesses

    #  per-layer FF loss
    def layer_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        l = self.goodness(z) - self.theta
        loss = -torch.mean(
            labels * F.logsigmoid(l) +
            (1 - labels) * F.logsigmoid(-l)
        )
        return loss

    #  layer-wise FF forward (training only)
    def ff_forward(
        self,
        x: torch.Tensor, # (B, input_dim) 
        labels: torch.Tensor,
    ) -> list[torch.Tensor]:
        
        losses = []
        h = x
        for linear, act in zip(self.hidden_layers, self.activations):
            h_norm = self._l2_normalize(h) # normalize
            z = act(linear(h_norm)) # activation output
            losses.append(self.layer_loss(z, labels))

            h = z.detach() # cut graph
            if self.dropout is not None:
                h = self.dropout(h)
        return losses  # len == number of hidden layers

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"ForwardNet(\n"
            f"  dims       : {cfg.input_dim} → {cfg.hidden_dims}  (goodness-classified over {cfg.num_classes} labels)\n"
            f"  activation : {cfg.activation}\n"
            f"  dropout    : {cfg.dropout}\n"
            f"  norm       : L2 length-norm (eps={cfg.norm_eps})\n"
            f"  theta      : {cfg.theta}  (mean-square goodness)\n"
            f"  params     : {sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )

    # HOOKS
    def register_activation_hooks(self) -> dict:
        """
        capture each layer's post-activation output z
        populates self._activations during
        forward()/ff_forward() call
        """
        for i, act in enumerate(self.activations):
            def make_fwd_hook(name: str):
                def hook(module, inp, out):
                    self._activations[name] = out.detach()
                return hook

            handle = act.register_forward_hook(make_fwd_hook(f"z_layer{i}"))
            self._hooks.append(handle)

        return self._activations

    def register_gradient_hooks(self) -> dict:
        """
        capture gradient at each hidden Linear's output
        """
        for i, linear in enumerate(self.hidden_layers):
            def make_bwd_hook(name: str):
                def hook(module, grad_input, grad_output):
                    self._gradients[name] = grad_output[0].detach()
                return hook

            handle = linear.register_full_backward_hook(make_bwd_hook(f"grad_layer{i}"))
            self._hooks.append(handle)

        return self._gradients

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()