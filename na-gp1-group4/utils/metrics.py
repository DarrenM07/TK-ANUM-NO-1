import numpy as np

def norm1(A):
    A = np.asarray(A)
    if A.ndim == 1: return np.sum(np.abs(A))
    return np.max(np.sum(np.abs(A), axis=0))

def norm_inf(A):
    A = np.asarray(A)
    if A.ndim == 1: return np.max(np.abs(A))
    return np.max(np.sum(np.abs(A), axis=1))

def hager_1norm_condest(A, solve):
    """
    Estimate cond_1(A) ≈ ||A||_1 * ||A^{-1}||_1 (Hager-like iteration).
    `solve(M, b)` must be *your* solver (no np.linalg).
    Returns (cond_est, ainv_norm_est).
    """
    n = A.shape[0]
    x = np.ones(n) / n
    est_old = 0.0
    for _ in range(10):
        s = np.sign(x); s[s == 0] = 1.0
        # Solve A^T y = s
        y = solve(A.T, s)
        j = int(np.argmax(np.abs(y)))
        e = np.zeros(n); e[j] = 1.0
        # Solve A z = e_j
        z = solve(A, e)
        est = np.sum(np.abs(z))
        x = z
        if est <= est_old + 1e-12:
            break
        est_old = est
    a1 = norm1(A)
    return a1 * est_old, est_old
