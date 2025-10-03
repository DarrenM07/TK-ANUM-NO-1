import os, csv, numpy as np
from utils.builders import random_distinct_nodes
from utils.timing import timer
from utils.metrics import norm_inf, hager_1norm_condest
from algorithms.vandermonde import build_vandermonde, solve_vandermonde_bjorck_pereyra
from algorithms.lu import solve_lu, lu_decompose, forward_substitution, back_substitution

SIZES = [5, 10, 25, 50, 100, 500, 1000]

def ensure_out():
    outdir = os.path.join("outputs", "vandermonde")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _cond1_estimate(A):
    def solve_with_lu(M, rhs):
        P,L,U = lu_decompose(M)
        y = forward_substitution(L, P @ rhs)
        return back_substitution(U, y)
    cond1, _ = hager_1norm_condest(A, solve_with_lu)
    return cond1

def run():
    out = ensure_out()
    rows = [("N","time_LU_ms","time_BP_ms","err_LU_inf","err_BP_inf","cond1_est")]
    for N in SIZES:
        x = random_distinct_nodes(N, seed=N)
        V = build_vandermonde(x)
        c_true = np.ones(N)
        b = V @ c_true

        # LU baseline
        with timer() as t:
            c_lu = solve_lu(V, b)
        t_lu = t()*1000
        err_lu = norm_inf(c_lu - c_true)

        # Efficient Vandermonde (Björck–Pereyra)
        with timer() as t:
            c_bp = solve_vandermonde_bjorck_pereyra(x, b)
        t_bp = t()*1000
        err_bp = norm_inf(c_bp - c_true)

        cond1 = _cond1_estimate(V)
        rows.append((N, t_lu, t_bp, err_lu, err_bp, cond1))

        print(f"[vandermonde] N={N:4d} | LU {t_lu:8.2f} ms, BP {t_bp:8.2f} ms | "
              f"err LU {err_lu:.2e}, BP {err_bp:.2e} | cond1~{cond1:.2e}")

    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("Saved CSV to", os.path.join(out, "results.csv"))

if __name__ == "__main__":
    run()
