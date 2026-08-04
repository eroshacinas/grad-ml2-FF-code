from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


# CONFIG

@dataclass
class DenseNetConfig:
    input_dim:    int         = 784
    hidden_dims:  List[int]   = field(default_factory=lambda: [500, 500, 500])
    num_classes:  int         = 10
    activation:   str         = "relu"
    dropout:      float       = 0.0


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class DenseNet(nn.Module):
    def __init__(self, cfg: DenseNetConfig = DenseNetConfig()):
        super().__init__()
        self.cfg = cfg

        act_cls = _ACTIVATIONS.get(cfg.activation)
        if act_cls is None:
            raise ValueError(f"Unknown activation '{cfg.activation}'. Choose from {list(_ACTIVATIONS)}")

        layers: List[nn.Module] = []
        # set initial in_dim to mnist input size: 28x28 = 784
        in_dim = cfg.input_dim

        for out_dim in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_cls()) # relu, tanh, gelu

            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))

            # set in_dim for next layer to current out_dim
            in_dim = out_dim

        # add output layer
        layers.append(nn.Linear(in_dim, cfg.num_classes))

        # stack all layers
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # nice print
    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"DenseNet(\n"
            f"  dims       : {cfg.input_dim} → {cfg.hidden_dims} → {cfg.num_classes}\n"
            f"  activation : {cfg.activation}\n"
            f"  dropout    : {cfg.dropout}\n"
            f"  params     : {sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )
    


    # hooks to capture intermediate activations and gradient flow
    def register_activation_hooks(self) -> dict:
        """
        attach forward hooks to every linear and activation layer
        returns the activations dict (populated after each forward pass).
        """
        self._activations: dict[str, torch.Tensor] = {}
        self._hooks = []

        for i, layer in enumerate(self.net):
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.Tanh, nn.GELU)):
                def make_fwd_hook(name: str):
                    def hook(module, input, output):
                        self._activations[name] = output.detach()
                    return hook

                name = f"{layer.__class__.__name__}_layer{i}"
                handle = layer.register_forward_hook(make_fwd_hook(name))
                self._hooks.append(handle)

        return self._activations

    def register_gradient_hooks(self) -> dict:
        """
        attach backward hooks to every linear and activation layer.
        returns the gradients dict (populated after each backward pass).
        """
        self._gradients: dict[str, torch.Tensor] = {}

        for i, layer in enumerate(self.net):
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.Tanh, nn.GELU)):
                def make_bwd_hook(name: str):
                    def hook(module, grad_input, grad_output):
                        self._gradients[name] = grad_output[0].detach()
                    return hook

                name = f"{layer.__class__.__name__}_layer{i}"
                handle = layer.register_full_backward_hook(make_bwd_hook(name))
                self._hooks.append(handle)

        return self._gradients

    def remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()