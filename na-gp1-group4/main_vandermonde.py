import csv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Impor fungsi dan modul yang sudah dibuat dalam folder proyek
from algorithms.lu import lu_decompose, solve_lu
from algorithms.vandermonde import divided_differences, newton_poly_eval_coefs
from utils.builders import build_vandermonde
from utils.metrics import compute_condition_number, relative_error
from utils.complexity import estimate_flops, estimate_memory_bytes

# KONFIGURASI EKSPERIMEN
OUTPUT_PATH = "outputs/vandermonde/results_with_chebyshev.csv"  # Lokasi file hasil eksperimen
N_VALUES = [5, 10, 25, 50, 100, 500, 1000]  # Ukuran matriks yang diuji
NODE_TYPES = ["equispaced", "chebyshev"]    # Jenis distribusi titik (node)

# FUNGSI BANTU: Pengukur waktu eksekusi
def measure_time(func, *args, **kwargs):
    """
    Mengukur lama waktu eksekusi sebuah fungsi (dalam milidetik).
    Digunakan untuk mencatat performa setiap solver.
    """
    start = time.perf_counter()               # Waktu mulai (presisi tinggi)
    result = func(*args, **kwargs)            # Jalankan fungsi yang diukur
    end = time.perf_counter()                 # Waktu akhir
    return result, (end - start) * 1000.0     # Kembalikan hasil dan waktu dalam milidetik

# MAIN EXPERIMENT
def run_vandermonde_experiment():
    rows = [] # Menyimpan hasil seluruh eksperimen dalam bentuk list baris
    header = [
        "nodes", "n", "solver", "time_ms", "rel_err",
        "cond1_est", "flops_model", "mem_model_bytes"
    ]

    # Loop melalui setiap jenis node dan setiap ukuran n
    for node_type in NODE_TYPES:
        for n in N_VALUES:
            print(f"Running experiment for {node_type} nodes, n={n} ...")

            # Bangun titik-titik node (equispaced atau Chebyshev)
            if node_type == "equispaced":
                xs = np.linspace(-1, 1, n) # Titik berjarak sama
            elif node_type == "chebyshev":
                # Titik Chebyshev tersebar lebih rapat di tepi interval
                xs = np.cos(np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n))

            # Bangun sistem Vandermonde V * coef = b
            true_coef = np.random.default_rng(42).uniform(-1, 1, size=n)  # Koefisien acak untuk polinomial
            V = build_vandermonde(xs)                                     # Matriks Vandermonde
            b = V @ true_coef                                             # Hitung nilai f(x) = V*c
            cond_est = compute_condition_number(V)                        # Hitung condition number (κ₁(V))

            # Solver 1: LU Decomposition
            (L, U, P), time_lu_ms = measure_time(lu_decompose, V)         # Dekomposisi LU
            x_lu, time_solve_ms = measure_time(solve_lu, V, b)            # Penyelesaian sistem LU
            total_time_lu = time_lu_ms + time_solve_ms                    # Total waktu = faktorisasi + solusi

            err_rel_lu = relative_error(V @ x_lu, b)                      # Error relatif hasil LU
            flops_lu = estimate_flops("lu", n)                            # Estimasi FLOPs
            mem_lu = estimate_memory_bytes("lu", n)                       # Estimasi memori

            rows.append([
                node_type, n, "LU (ours, P A = L U)",
                total_time_lu, err_rel_lu, cond_est, flops_lu, mem_lu
            ])

            # Solver 2: Newton / Divided Differences
            coef_newton, time_newton_ms = measure_time(divided_differences, xs, b)
            poly_coef, time_poly_ms = measure_time(newton_poly_eval_coefs, xs, coef_newton)
            total_time_newton = time_newton_ms + time_poly_ms

            err_rel_newton = relative_error(V @ poly_coef, b)             # Error relatif Newton
            flops_newton = estimate_flops("newton", n)                    # Estimasi FLOPs
            mem_newton = estimate_memory_bytes("newton", n)               # Estimasi memori

            rows.append([
                node_type, n, "Newton/DivDiff",
                total_time_newton, err_rel_newton, cond_est, flops_newton, mem_newton
            ])

    # Simpan hasil ke file CSV
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)       # Tulis header kolom
        writer.writerows(rows)        # Tulis semua baris hasil

    print(f"\n Results saved to {OUTPUT_PATH}")

    # Konversi hasil ke DataFrame untuk analisis dan plotting
    df = pd.DataFrame(rows, columns=header)

    # Plot: Runtime vs n (skala log)
    plt.figure(figsize=(8, 5))
    for node_type in NODE_TYPES:
        for solver, style in zip(["LU (ours, P A = L U)", "Newton/DivDiff"], ["o-", "x--"]):
            subset = df[(df["nodes"] == node_type) & (df["solver"] == solver)]
            plt.plot(subset["n"], subset["time_ms"], style, label=f"{solver} - {node_type}")

    plt.xscale("log")                               # Sumbu X dalam log-scale
    plt.yscale("log")                               # Sumbu Y dalam log-scale
    plt.xlabel("Matrix size n (log scale)")         # Label sumbu X
    plt.ylabel("Runtime (ms, log scale)")           # Label sumbu Y
    plt.title("Runtime Comparison (LU vs Newton, Equispaced vs Chebyshev)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/vandermonde/runtime_comparison.png")  # Simpan ke file
    plt.show()

    # Plot: Relative Error vs n (skala log)
    plt.figure(figsize=(8, 5))
    for node_type in NODE_TYPES:
        for solver, style in zip(["LU (ours, P A = L U)", "Newton/DivDiff"], ["o-", "x--"]):
            subset = df[(df["nodes"] == node_type) & (df["solver"] == solver)]
            plt.plot(subset["n"], subset["rel_err"], style, label=f"{solver} - {node_type}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Matrix size n (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.title("Relative Error Comparison (LU vs Newton, Equispaced vs Chebyshev)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/vandermonde/error_comparison.png")
    plt.show()

    # INTERPRETASI NUMERIK
    print("""
 Interpretasi Numerik: Mengapa Vandermonde Ill-Conditioned

1️ Matriks Vandermonde memiliki elemen V[i,j] = x_i^(j-1).
   Saat n membesar dan x_i merata (equispaced), kolom-kolom matriks menjadi hampir linear dependent.

2️ Determinan Vandermonde = ∏(x_j - x_i).
   Untuk node equispaced dengan jarak kecil, determinan ini mendekati nol,
   sehingga condition number (κ(V)) menjadi sangat besar → sistem menjadi sangat sensitif.

3️ Akibatnya, error pembulatan kecil pada b dapat menyebabkan error besar pada solusi x,
   terutama untuk Newton/DivDiff.

4️ Chebyshev nodes (x_i = cos((2i−1)/(2n)π)) menyebarkan titik lebih rapat di ujung interval,
   membuat kolom Vandermonde lebih ortogonal dan menjaga determinan lebih besar.

Kesimpulan:
- LU decomposition: stabil tapi lambat (O(n³))
- Newton/DivDiff: cepat (O(n²)) tapi lebih rentan terhadap ill-conditioning
- Chebyshev nodes: meningkatkan kestabilan numerik secara signifikan tanpa mengorbankan efisiensi.
""")

# MAIN
if __name__ == "__main__":
    run_vandermonde_experiment()
