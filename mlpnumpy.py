import numpy as np

HIDDEN_SIZES   = [256, 256]
MAX_GRAD_NORM  = 1.0


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class LayerNorm:
    """Einfache Layer‑Norm (pro Feature) für NumPy."""

    def __init__(self, size: int, eps: float = 1e-5):
        self.gamma = np.ones(size)
        self.beta  = np.zeros(size)
        self.eps   = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mu  = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        return self.gamma * (x - mu) / np.sqrt(var + self.eps) + self.beta


class DeepMLP:
    """Mehrschichtiges Perzeptron + LayerNorm + Adam (NumPy only)."""

    def __init__(self, inp: int, hidden: list[int], out: int):
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.norms: list[LayerNorm] = []
        sizes = [inp] + hidden + [out]
        rng = np.random.default_rng(42)

        for i in range(len(sizes) - 1):
            w = rng.standard_normal((sizes[i], sizes[i + 1])) / np.sqrt(sizes[i])
            b = np.zeros(sizes[i + 1])
            self.W.append(w)
            self.b.append(b)
            if i < len(sizes) - 2:
                self.norms.append(LayerNorm(sizes[i + 1]))

        self.zero_grad()
        # Adam‑Momente
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t  = 0

    # ---------------- FWD / BWD ----------------
    def zero_grad(self):
        self.dW = [np.zeros_like(w) for w in self.W]
        self.db = [np.zeros_like(b) for b in self.b]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = [x]
        h = x
        for i in range(len(self.W) - 1):
            h = relu(self.norms[i](h @ self.W[i] + self.b[i]))
            self.x.append(h)
        out = h @ self.W[-1] + self.b[-1]
        self.x.append(out)  # convenience
        return out

    def backward(self, dout: np.ndarray):
        # dout entspricht dL/d(out)
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:            # Ableitung ReLU
                dout = dout * (self.x[i + 1] > 0)
            self.dW[i] += self.x[i].T @ dout
            self.db[i] += dout.sum(axis=0)
            dout = dout @ self.W[i].T

    # ------------- Adam Step -------------
    def step(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.t += 1
        beta1t = 1 - b1 ** self.t
        beta2t = 1 - b2 ** self.t

        # Global Grad‑Clipping
        g_norm = np.sqrt(sum((dw ** 2).sum() + (db ** 2).sum() for dw, db in zip(self.dW, self.db)))
        scale = min(1.0, MAX_GRAD_NORM / (g_norm + 1e-8))

        for i in range(len(self.W)):
            dw = self.dW[i] * scale
            db = self.db[i] * scale

            self.mW[i] = b1 * self.mW[i] + (1 - b1) * dw
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * (dw ** 2)
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * db
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * (db ** 2)

            m_hat_w = self.mW[i] / beta1t
            v_hat_w = self.vW[i] / beta2t
            m_hat_b = self.mb[i] / beta1t
            v_hat_b = self.vb[i] / beta2t

            self.W[i] -= lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
            self.b[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        self.zero_grad()
