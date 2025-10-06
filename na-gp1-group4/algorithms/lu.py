import numpy as np # Mengimpor library numpy untuk operasi numerik matriks dan vektor

def lu_decompose(A):
    """
    Melakukan dekomposisi LU dengan pivot parsial (Doolittle Method)
    Sehingga diperoleh P @ A = L @ U
    - L = matriks lower triangular dengan diagonal utama bernilai 1
    - U = matriks upper triangular
    - P = matriks pivot (permutasi baris)
    """
    A = A.copy().astype(float)      # Membuat salinan matriks A dan mengubah tipe datanya menjadi float
    n = A.shape[0]                  # Menyimpan ukuran matriks (jumlah baris = kolom)
    P = np.eye(n)                   # Membuat matriks identitas n×n untuk matriks pivot P
    L = np.zeros_like(A)            # Inisialisasi matriks L berukuran sama dengan A berisi nol
    U = A.copy()                    # Menyalin matriks A ke U (karena U akan dimodifikasi selama eliminasi)\
    # Loop utama untuk setiap kolom k (proses eliminasi Gaussian)
    for k in range(n):
        # Menentukan baris pivot dengan mencari nilai maksimum pada kolom k mulai dari baris k
        pivot = np.argmax(np.abs(U[k:, k])) + k
        # Jika elemen pivot bernilai nol, matriks singular (tidak bisa didekomposisi)
        if U[pivot, k] == 0:
            raise ValueError("Singular matrix in LU.")
        # Jika baris pivot bukan baris k, maka tukar baris (partial pivoting)
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                # Jika sudah ada kolom sebelumnya di L, ikut ditukar juga bagian bawahnya
                L[[k, pivot], :k] = L[[pivot, k], :k]
        # Set diagonal utama L ke 1 (sesuai metode Doolittle)
        L[k, k] = 1.0
        # Proses eliminasi: membuat elemen di bawah pivot menjadi nol
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    # Mengembalikan matriks pivot (P), lower (L), dan upper (U)
    return P, L, U

def forward_substitution(L, b):
    """
    Menyelesaikan sistem L*y = b
    dengan metode substitusi maju (forward substitution)
    L adalah matriks segitiga bawah (lower triangular)
    """
    n = L.shape[0]               # Ukuran matriks (jumlah baris)
    y = np.zeros(n)              # Inisialisasi vektor solusi y dengan nol
    for i in range(n):           # Iterasi dari baris atas ke bawah
        # Menghitung nilai y[i] berdasarkan nilai sebelumnya
        y[i] = b[i] - np.dot(L[i, :i], y[:i])  # b[i] dikurangi hasil kali elemen-elemen L dengan y sebelumnya
    return y                     # Mengembalikan hasil vektor y

def back_substitution(U, y):
    """
    Menyelesaikan sistem U*x = y
    dengan metode substitusi mundur (back substitution)
    U adalah matriks segitiga atas (upper triangular)
    """
    n = U.shape[0]               # Ukuran matriks (jumlah baris)
    x = np.zeros(n)              # Inisialisasi vektor solusi x dengan nol
    # Iterasi dari baris bawah ke atas
    for i in range(n - 1, -1, -1):
        # Menghitung jumlah elemen di kanan diagonal (dot product U[i, i+1:] dan x[i+1:])
        s = y[i] - np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = s / U[i, i]       # Bagi dengan elemen diagonal untuk mendapatkan x[i]
    return x                     # Mengembalikan hasil vektor x

def solve_lu(A, b):
    """
    Menyelesaikan sistem linear A*x = b menggunakan dekomposisi LU dengan pivot parsial
    Langkah:
    1. Lakukan dekomposisi LU → P, L, U
    2. Hitung pb = P @ b (karena P*A = L*U)
    3. Selesaikan L*y = P*b dengan forward substitution
    4. Selesaikan U*x = y dengan back substitution
    """
    P, L, U = lu_decompose(A)   # Langkah 1: dekomposisi LU dengan pivot parsial
    pb = P @ b                  # Langkah 2: kalikan b dengan matriks pivot P
    y = forward_substitution(L, pb)  # Langkah 3: cari y dari L*y = P*b
    return back_substitution(U, y)   # Langkah 4: cari x dari U*x = y