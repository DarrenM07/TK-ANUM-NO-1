import numpy as np

def lu_decompose(A):
    """Doolittle LU with partial pivoting; P@A=L@U; L has unit diagonal."""
    A = A.copy().astype(float)
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros_like(A)
    U = A.copy()
    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if U[pivot, k] == 0:
            raise ValueError("Singular matrix in LU.")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        L[k, k] = 1.0
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return P, L, U

def forward_substitution(L, b):
    n = L.shape[0]; y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def back_substitution(U, y):
    n = U.shape[0]; x = np.zeros(n)
    for i in range(n-1, -1, -1):
        s = y[i] - np.dot(U[i, i+1:], x[i+1:])
        x[i] = s / U[i, i]
    return x

def solve_lu(A, b):
    P, L, U = lu_decompose(A)
    pb = P @ b
    y = forward_substitution(L, pb)
    return back_substitution(U, y)
