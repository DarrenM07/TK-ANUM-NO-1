import numpy as np

def random_distinct_nodes(n, seed=0):
    """
    Membuat himpunan titik x (nodes) acak yang berbeda (distinct) di interval [-1, 1].
    Digunakan untuk eksperimen interpolasi agar titik-titik tidak saling berdekatan atau duplikat.
    """

    rng = np.random.default_rng(seed)        # Membuat generator bilangan acak (reproducible dengan seed)
    x = rng.uniform(-1, 1, size=n)           # Mengambil n angka acak dari interval [-1, 1]
    x.sort()                                 # Urutkan titik-titik agar berurutan dari kecil ke besar

    # Koreksi titik-titik yang terlalu berdekatan (jarak < 1e-3)
    for i in range(1, n):
        if abs(x[i] - x[i - 1]) < 1e-3:      # Jika dua titik hampir sama
            x[i] += (i + 1) * 1e-3           # Geser titik ke kanan sedikit agar berbeda

    return x                                 # Kembalikan array nodes yang sudah unik dan terurut


def build_vandermonde(xs):
    """
    Membangun matriks Vandermonde dengan basis monomial standar.
    Matriks ini digunakan untuk sistem interpolasi polinomial:
        V * c = f
    di mana:
        V[i, j] = (x_i)^j
    dengan:
        - x_i adalah titik data (node)
        - j adalah pangkat polinomial (0 sampai n-1)
    
    Contoh:
    Jika xs = [1, 2, 3], maka:
        V =
        [[1, 1, 1],
         [1, 2, 4],
         [1, 3, 9]]
    """

    xs = np.array(xs, dtype=float)           # Pastikan xs berupa array float (bukan list)
    n = len(xs)                              # Jumlah titik data (node)
    V = np.zeros((n, n), dtype=float)        # Inisialisasi matriks V berukuran n×n dengan nol

    # Isi setiap elemen matriks V[i, j] = x_i^j
    for i in range(n):                       # Iterasi untuk setiap baris (setiap node)
        for j in range(n):                   # Iterasi untuk setiap kolom (setiap pangkat)
            V[i, j] = xs[i] ** j             # Hitung nilai pangkat dan simpan di V
    return V                                 # Kembalikan matriks Vandermonde hasil konstruksi