import numpy as np

def build_vandermonde(x):
    """Vandermonde with basis [1, x, x^2, ..., x^{n-1}]."""
    x = np.asarray(x, dtype=float)
    n = x.size
    V = np.ones((n, n), dtype=float)
    for j in range(1, n):
        V[:, j] = V[:, j-1] * x
    return V

def solve_vandermonde_bjorck_pereyra(x, b):
    """Solve V(x)c = b using O(n^2) Björck–Pereyra (distinct nodes)."""
    x = np.asarray(x, dtype=float)
    b = np.asarray(b, dtype=float).copy()
    n = x.size
    # forward sweep
    for k in range(n-1):
        denom = x[k+1:] - x[k]
        if np.any(np.isclose(denom, 0)):
            raise ValueError("Duplicate nodes.")
        b[k+1:] = (b[k+1:] - b[k]) / denom
    # backward sweep
    c = np.zeros(n, dtype=float)
    c[-1] = b[-1]
    for k in range(n-2, -1, -1):
        c[k] = b[k] - x[k] * c[k+1]
        for j in range(k+1, n-1):
            c[j] = c[j] - x[k] * c[j+1]
    return c
