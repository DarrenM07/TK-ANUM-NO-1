import numpy as np  # Mengimpor numpy untuk operasi numerik matriks dan vektor

def norm1(A):
    """
    Menghitung norma-1 dari matriks atau vektor A.
    - Jika A adalah vektor: jumlah nilai absolut dari semua elemen.
    - Jika A adalah matriks: maksimum dari jumlah absolut setiap kolom.
    """
    A = np.asarray(A)                            # Pastikan A berupa array numpy
    if A.ndim == 1:                              # Jika A adalah vektor 1D
        return np.sum(np.abs(A))                 # Kembalikan jumlah nilai absolut
    return np.max(np.sum(np.abs(A), axis=0))     # Jika matriks: ambil maksimum dari jumlah per kolom


def norm_inf(A):
    """
    Menghitung norma-infinity dari matriks atau vektor A.
    - Jika A adalah vektor: nilai absolut maksimum.
    - Jika A adalah matriks: maksimum dari jumlah absolut setiap baris.
    """
    A = np.asarray(A)                            # Pastikan A berupa array numpy
    if A.ndim == 1:                              # Jika A adalah vektor
        return np.max(np.abs(A))                 # Nilai absolut maksimum
    return np.max(np.sum(np.abs(A), axis=1))     # Jika matriks: maksimum dari jumlah per baris


def hager_1norm_condest(A, solve):
    """
    Mengestimasi condition number 1-norm (cond_1(A)) dengan algoritma iteratif mirip Hager (1984).

    Pendekatan ini menghindari perhitungan A^{-1} secara eksplisit.
    Estimasi dilakukan berdasarkan:
        cond_1(A) ≈ ||A||_1 * ||A^{-1}||_1

    Parameter:
    - A : ndarray
        Matriks persegi yang ingin dihitung kondisi numeriknya.
    - solve : callable
        Fungsi pemecah sistem linear (custom solver) yang bisa menyelesaikan A x = b.
        Misalnya, solve = lambda A, b: np.linalg.solve(A, b)

    Return:
    - cond_est : perkiraan condition number berdasarkan norma-1.
    - ainv_norm_est : perkiraan norma-1 dari A^{-1}.
    """
    n = A.shape[0]                               # Ukuran matriks (jumlah baris)
    x = np.ones(n) / n                           # Inisialisasi vektor awal x = [1/n, 1/n, ..., 1/n]
    est_old = 0.0                                # Nilai estimasi sebelumnya (untuk konvergensi)

    # Iterasi maksimum 10 kali (biasanya cukup untuk konvergen)
    for _ in range(10):
        s = np.sign(x)                           # Ambil tanda dari setiap elemen x (+1 atau -1)
        s[s == 0] = 1.0                          # Jika ada nilai nol, ubah menjadi +1

        # Selesaikan sistem A^T * y = s
        y = solve(A.T, s)

        # Pilih indeks j di mana |y_j| maksimum
        j = int(np.argmax(np.abs(y)))

        # Bentuk vektor satuan e_j (semua nol kecuali 1 di posisi j)
        e = np.zeros(n)
        e[j] = 1.0

        # Selesaikan sistem A * z = e_j → z ≈ kolom j dari A^{-1}
        z = solve(A, e)

        # Estimasi norma 1 dari A^{-1} sebagai jumlah nilai absolut dari z
        est = np.sum(np.abs(z))

        # Update x untuk iterasi berikutnya
        x = z

        # Cek konvergensi: jika tidak ada peningkatan signifikan, hentikan
        if est <= est_old + 1e-12:
            break

        est_old = est                            # Simpan estimasi terakhir

    a1 = norm1(A)                                # Hitung norma-1 dari A
    return a1 * est_old, est_old                 # Kembalikan (cond_1(A) ~ ||A||_1 * ||A^{-1}||_1, ||A^{-1}||_1)


def relative_error(pred, true):
    """
    Menghitung error relatif antara dua vektor atau hasil prediksi.

    Formula:
        rel_err = ||pred - true|| / ||true||
    Di mana ||.|| adalah norma Euclidean (L2 norm).
    """
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


def compute_condition_number(A):
    """
    Menghitung condition number (κ) dari matriks A dengan menggunakan norma-1.

    Secara matematis:
        κ₁(A) = ||A||₁ * ||A⁻¹||₁

    Fungsi ini menggunakan implementasi built-in NumPy (np.linalg.cond)
    yang secara default menghitung nilai singular terbesar / terkecil dari A.
    """
    return np.linalg.cond(A, 1)                  # Menggunakan norma-1 untuk menghitung condition number
