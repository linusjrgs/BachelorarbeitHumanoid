# =========================
# utils.py – helper classes & functions (PyTorch only)
# =========================
import torch

__all__ = [
    "RunningNorm",
    "safe_atanh",
    "explained_variance",
]


def safe_atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically safe arctanh used for action squashing/unsquashing.

    Clamps the input to the open interval (‑1, 1) before applying `torch.atanh`.
    """
    return torch.atanh(x.clamp(-1 + eps, 1 - eps))


class RunningNorm:
    """Track running mean/variance across multiple workers."""

    def __init__(self, shape, eps: float = 1e-4, device: str = "cpu"):
        self.device = device
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = eps

    def update(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.mean.dtype, device=self.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.size(0)
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot

    def norm(self, x) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=self.mean.dtype, device=self.device)
        return (x - self.mean) / (torch.sqrt(self.var) + 1e-8)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    if not torch.is_tensor(y_pred):
        y_pred = torch.as_tensor(y_pred)
    if not torch.is_tensor(y_true):
        y_true = torch.as_tensor(y_true)
    var_y = torch.var(y_true)
    if var_y == 0:
        return float("nan")
    return float((1 - torch.var(y_true - y_pred) / var_y).item())
