"""
In diesem Modul befinden sich kleine Hilfsfunktionen und eine Klasse, 
die während des Trainingsprozesses im Reinforcement Learning nützlich sind.

Enthalten:
- Numerisch sichere Berechnung von arctanh (für Actionscaling)
- Laufende Schätzung von Mittelwert und Varianz (für Normalisierung)
- Berechnung der "Explained Variance" (zur Bewertung von Value-Funktionen)

Alle Funktionen sind auf PyTorch ausgelegt und GPU-kompatibel.
"""

import torch

# __all__ listet alle exportierten Funktionen/Klassen → wird bei `from utils import *` verwendet
__all__ = [
    "RunningNorm",
    "safe_atanh",
    "explained_variance",
]


def safe_atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Numerisch sichere Variante von arctanh (inverse Hyperbolische Tangens).

    Anwendung: beim "Squashen" oder "Unsquashen" von Aktionen.
    → Viele RL-Algorithmen begrenzen Actions in [-1,1]. Um mathematische Fehler zu vermeiden,
    wird der Wertebereich hier leicht beschnitten (clamped), bevor arctanh angewendet wird.

    Args:
        x (torch.Tensor): Eingabewerte (typisch: Actions)
        eps (float): Sicherheitsabstand zu den Grenzen -1/1

    Returns:
        torch.Tensor: stabil berechnetes arctanh(x)
    """
    return torch.atanh(x.clamp(-1 + eps, 1 - eps))


class RunningNorm:
    """
    Klasse zur laufenden Schätzung von Mittelwert und Varianz.

    → Wichtig beim Normalisieren von Beobachtungen (Observations) oder Belohnungen.
    → Durch online Update geeignet für kontinuierliches Training und parallele Workers.

    Args:
        shape: Form der beobachteten Daten
        eps: kleiner Startwert für count (vermeidet Division durch 0)
        device: CPU oder GPU
    """

    def __init__(self, shape, eps: float = 1e-4, device: str = "cpu"):
        # initialisiere Mittelwert und Varianz
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = eps

    def update(self, x: torch.Tensor):
        """
        Aktualisiert Mittelwert und Varianz basierend auf einer neuen Batch von Daten.

        → wichtig bei verteiltem Training (mehrere Batches von verschiedenen Prozessen).

        Args:
            x (torch.Tensor): Neue Beobachtungen (Batch)
        """
        # stelle sicher, dass x ein Tensor ist
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.mean.dtype, device=self.device)

        # Batch-Statistiken berechnen
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)

        # Differenz zur aktuellen Mittelwertschätzung
        delta = batch_mean - self.mean
        tot = self.count + batch_count

        # neuen Mittelwert berechnen (gewichtetes Mittel)
        new_mean = self.mean + delta * batch_count / tot

        # neue Varianz berechnen (Welford's Algorithmus)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot

        # update speichern
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def norm(self, x) -> torch.Tensor:
        """
        Normalisiert Eingabewerte basierend auf aktueller Mittelwert-/Varianzschätzung.

        Typischer Anwendungsfall:
        - Normalisierung von Observations → stabileres Lernen

        Args:
            x (torch.Tensor oder Array): Zu normalisierende Daten

        Returns:
            torch.Tensor: Normalisierte Daten
        """
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.mean.dtype, device=self.device)

        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Berechnet die Explained Variance zwischen vorhergesagten und echten Werten.

    Anwendung: zur Bewertung der Qualität einer Value-Funktion.
    → Wert nahe 1.0: sehr gute Vorhersage
    → Wert 0.0: keine Verbesserung gegenüber konstantem Mittelwert
    → Wert <0: Modell verschlechtert Vorhersage im Vergleich zum Mittelwert

    Formel:
    EV = 1 - Var(y_true - y_pred) / Var(y_true)

    Args:
        y_pred (torch.Tensor): vorhergesagte Werte (z.B. Value Function Outputs)
        y_true (torch.Tensor): tatsächliche Zielwerte (Returns / Targets)

    Returns:
        float: Explained Variance (zwischen -∞ und 1)
    """
    # absichern, dass beide Tensoren sind
    if not torch.is_tensor(y_pred):
        y_pred = torch.as_tensor(y_pred)
    if not torch.is_tensor(y_true):
        y_true = torch.as_tensor(y_true)

    # Varianz der echten Werte berechnen
    var_y = torch.var(y_true)
    if var_y == 0:
        return float("nan")  # wenn y_true konstant → EV undefiniert

    # Explained Variance berechnen und als float zurückgeben
    return float((1 - torch.var(y_true - y_pred) / var_y).item())
