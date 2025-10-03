def flops_lu(n):                 # ~ (2/3) n^3 for factorization + ~2 n^2 for triangular solves
    return (2.0/3.0)*n**3 + 2.0*n**2
def memory_lu(n):                # ≈ 3 n^2 doubles (A/L/U dense) — coarse upper bound
    return 3*n*n
def flops_bjorck_pereyra(n):     # O(n^2) forward+backward sweeps
    return 2.0*n**2
def memory_bjorck_pereyra(n):    # x,b,c vectors
    return 3*n
