# =========================
# mlp.py – Flexible MLP definition
# =========================
import torch
import torch.nn as nn

MAX_GRAD_NORM = 1.0


class MLP(nn.Module):
    """Simple feed‑forward network with optional output activation.

    Args:
        input_dim:  dimensionality of input features
        hidden_sizes: list of hidden layer sizes
        output_dim: dimensionality of network output
        final_activation: torch.nn.Module inserted **after** the last linear
            layer (e.g. ``nn.Tanh()``).  Pass ``None`` for *no* activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        output_dim: int,
        *,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.LayerNorm(h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, output_dim))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)
