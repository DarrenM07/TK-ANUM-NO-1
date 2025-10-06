import numpy as np

def build_vandermonde(x):
    """
    Membentuk matriks Vandermonde berdasarkan titik x.
    Basis yang digunakan adalah [1, x, x^2, ..., x^(n-1)].

    Contoh:
    Jika x = [1, 2, 3],
    maka V =
    [[1, 1, 1],
     [1, 2, 4],
     [1, 3, 9]]
    """
    x = np.asarray(x, dtype=float)          # Pastikan x berupa array float (bukan list)
    n = x.size                              # Banyaknya titik (jumlah kolom & baris)
    V = np.ones((n, n), dtype=float)        # Inisialisasi matriks dengan semua elemen = 1
    for j in range(1, n):                   # Iterasi dari kolom ke-2 sampai terakhir
        V[:, j] = V[:, j - 1] * x           # Setiap kolom j = kolom sebelumnya dikali x (rekursif)
    return V                                # Kembalikan matriks Vandermonde

def solve_vandermonde_bjorck_pereyra(x, b):
    """
    Menyelesaikan sistem V(x) * c = b dengan algoritma Björck–Pereyra (kompleksitas O(n^2)).
    Algoritma ini efisien untuk node x yang berbeda (distinct nodes).
    Hasilnya adalah koefisien c dari polinomial interpolasi.

    Metode ini lebih stabil dan cepat dibanding langsung melakukan LU pada Vandermonde.
    """
    x = np.asarray(x, dtype=float)          # Pastikan x berupa array float
    b = np.asarray(b, dtype=float).copy()   # Salin b agar tidak mengubah array aslinya
    n = x.size                              # Banyaknya titik data

    # --------------------------
    # Forward sweep (devided differences)
    # --------------------------
    for k in range(n - 1):                  # Iterasi dari k=0 hingga n-2
        denom = x[k + 1:] - x[k]            # Selisih antar titik x
        if np.any(np.isclose(denom, 0)):    # Jika ada titik yang sama, hentikan (duplikat node)
            raise ValueError("Duplicate nodes.")
        b[k + 1:] = (b[k + 1:] - b[k]) / denom  # Update b sesuai formula Björck–Pereyra

    # --------------------------
    # Backward sweep (rekonstruksi koefisien)
    # --------------------------
    c = np.zeros(n, dtype=float)            # Inisialisasi array hasil koefisien
    c[-1] = b[-1]                           # Koefisien terakhir langsung diambil dari b
    for k in range(n - 2, -1, -1):          # Iterasi mundur dari n-2 ke 0
        c[k] = b[k] - x[k] * c[k + 1]       # Hitung c[k] berdasarkan x[k] dan c[k+1]
        for j in range(k + 1, n - 1):       # Koreksi elemen di antara (loop dalam)
            c[j] = c[j] - x[k] * c[j + 1]   # Update koefisien antar langkah
    return c                                # Kembalikan array koefisien hasil


def divided_differences(xs, ys):
    """
    Menghitung koefisien Newton (divided differences) untuk polinomial interpolasi
    melalui titik-titik (xs, ys).

    Hasilnya adalah array koefisien yang membentuk polinomial Newton:
    P(x) = c0 + c1(x - x0) + c2(x - x0)(x - x1) + ... + cn(x - x0)...(x - x_{n-1})
    """
    xs = np.array(xs, dtype=float)          # Pastikan xs berupa array float
    ys = np.array(ys, dtype=float)          # Pastikan ys berupa array float
    n = len(xs)                             # Banyaknya titik data
    coef = ys.copy()                        # Salin ys untuk memulai pembentukan koefisien

    # Iterasi untuk membentuk tabel divided differences
    for j in range(1, n):
        # Update baris ke-j: (f[x_j,...] - f[x_{j-1},...]) / (x_j - x_{j-1})
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (xs[j:n] - xs[0:n - j])
    return coef                             # Kembalikan koefisien Newton (divided difference)


def newton_poly_eval_coefs(xs, coef):
    """
    Mengubah koefisien Newton menjadi koefisien pada basis monomial standar.
    Artinya, mengonversi bentuk Newton ke bentuk polinomial biasa:
        P(x) = c0 + c1*x + c2*x^2 + ...
    Agar hasilnya bisa dibandingkan dengan solusi dari Vandermonde langsung.

    Input:
        xs    = titik data (node)
        coef  = koefisien dari bentuk Newton
    Output:
        poly  = koefisien bentuk monomial [c0, c1, c2, ...]
    """
    n = len(xs)                             # Jumlah titik data
    poly = np.array([coef[-1]], dtype=float)  # Mulai dari suku tertinggi (koefisien terakhir)

    # Iterasi mundur dari koefisien ke-(n-2) hingga 0
    for k in range(n - 2, -1, -1):
        shifted = np.zeros(len(poly) + 1, dtype=float)  # Membuat ruang untuk menambah derajat polinomial
        shifted[1:] = poly                              # Geser seluruh suku ke kanan (x * poly)
        shifted[:-1] -= xs[k] * poly                    # Hitung (x - x_k) * poly
        poly = shifted                                  # Update poly dengan hasil perkalian
        poly[0] += coef[k]                              # Tambahkan koefisien Newton ke suku konstan
    return poly                                         # Kembalikan koefisien monomial hasil konversi