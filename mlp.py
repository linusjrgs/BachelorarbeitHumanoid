"""
In diesem Modul wird ein flexibles Multi-Layer Perceptron (MLP) definiert.

Dieses einfache feedforward Netzwerk wird im Reinforcement Learning z.B. als:
- Policy-Netzwerk (für Aktionsausgabe)
- Value-Netzwerk (für Schätzung der Value-Funktion)

verwendet.

Features:
- beliebig viele Hidden-Layer (konfigurierbar über `hidden_sizes`)
- Layer-Normalisierung nach jeder Linearschicht (stabilisiert Training)
- optionale Aktivierungsfunktion **am Ausgang** (z.B. Tanh für Actionscaling)
"""

import torch
import torch.nn as nn

# Maximaler Gradienten-Norm (z.B. für Gradient Clipping beim Training)
MAX_GRAD_NORM = 1.0


class MLP(nn.Module):
    """
    Einfaches feedforward MLP mit optionaler Ausgangs-Aktivierung.

    Args:
        input_dim (int): Dimension der Eingabedaten
        hidden_sizes (list[int]): Liste der Hidden-Layer Größen
        output_dim (int): Dimension der Ausgabeschicht
        final_activation (nn.Module | None): Aktivierungsfunktion **nach** letzter Linearschicht
            (z.B. nn.Tanh() für Actionscaling). Wenn None → keine Aktivierung.

    Beispiel:
        MLP(10, [64, 64], 4, final_activation=nn.Tanh())
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

        # Liste für Layer-Module
        layers: list[nn.Module] = []

        # aktueller Input → wird von Layer zu Layer angepasst
        last = input_dim

        # Hidden-Layer bauen
        for h in hidden_sizes:
            layers += [
                nn.Linear(last, h),    # Lineare Transformation
                nn.LayerNorm(h),      # LayerNorm (hilft gegen instabile Aktivierungen)
                nn.ReLU(),            # ReLU Aktivierung
            ]
            last = h  # für nächste Schicht

        # letzte Linearschicht (Ausgabe)
        layers.append(nn.Linear(last, output_dim))

        # optionale Aktivierungsfunktion nach letzter Schicht
        if final_activation is not None:
            layers.append(final_activation)

        # alle Layer als Sequential-Modul speichern
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Netzwerk.

        Args:
            x (torch.Tensor): Eingabedaten

        Returns:
            torch.Tensor: Ausgabe des Netzwerks (z.B. Policy logits oder Value-Schätzungen)
        """
        return self.net(x)
