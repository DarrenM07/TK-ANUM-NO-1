def flops_lu(n):
    """
    Menghitung perkiraan jumlah operasi floating-point (FLOPs)
    untuk metode LU decomposition dengan penyelesaian sistem segitiga.

    Rumus perkiraan:
        ~ (2/3)*n^3  untuk proses faktorisasi A = L*U
        + ~ 2*n^2    untuk dua kali triangular solve (Ly = b dan Ux = y)
    """
    return (2.0 / 3.0) * n**3 + 2.0 * n**2


def memory_lu(n):
    """
    Menghitung estimasi jumlah elemen memori yang digunakan (dalam satuan 'jumlah elemen'),
    bukan byte, oleh metode LU.

    Perkiraan:
        ~ 3 * n^2 elemen (karena menyimpan matriks A, L, dan U yang semuanya berukuran n×n)
    """
    return 3 * n * n


def flops_bjorck_pereyra(n):
    """
    Menghitung estimasi jumlah operasi FLOPs untuk algoritma Björck–Pereyra.

    Kompleksitas algoritma ini adalah O(n^2), karena melibatkan
    forward dan backward sweeps (dua loop bersarang sederhana).
    """
    return 2.0 * n**2


def memory_bjorck_pereyra(n):
    """
    Menghitung estimasi penggunaan memori untuk algoritma Björck–Pereyra.

    Hanya membutuhkan tiga vektor utama:
        x (nodes), b (values), dan c (koefisien hasil),
    masing-masing berukuran n elemen.
    """
    return 3 * n


def estimate_flops(model: str, n: int) -> int:
    """
    Mengestimasi jumlah operasi floating-point (FLOPs) berdasarkan model solver yang digunakan.

    Parameter:
    - model : str
        Nama metode (misalnya 'LU', 'newton', 'vandermonde')
    - n : int
        Ukuran matriks atau banyaknya titik (nodes)

    Return:
    - int : estimasi jumlah operasi

    Logika:
    - Jika model dimulai dengan 'lu' → gunakan rumus O(n^3)
    - Jika mengandung kata 'newton' → gunakan O(n^2)
    - Jika bukan keduanya → asumsi umum O(n^3)
    """
    if model.lower().startswith("lu"):
        return (2 / 3) * n**3
    elif "newton" in model.lower():
        return n**2
    else:
        return n**3


def estimate_memory_bytes(model: str, n: int) -> int:
    """
    Mengestimasi jumlah penggunaan memori (dalam satuan byte)
    berdasarkan model solver yang digunakan.

    Parameter:
    - model : str
        Nama metode (misalnya 'LU', 'newton', dll)
    - n : int
        Ukuran matriks atau banyaknya titik

    Return:
    - int : estimasi penggunaan memori dalam byte

    Asumsi:
    - Tipe data float64 → 8 byte per elemen
    - LU membutuhkan n^2 elemen
    - Newton hanya butuh vektor ukuran n
    - Model lain dianggap seperti LU (n^2)
    """
    if model.lower().startswith("lu"):
        return n**2 * 8  # float64 = 8 byte per elemen
    elif "newton" in model.lower():
        return n * 8     # hanya vektor, bukan matriks penuh
    else:
        return n**2 * 8  # asumsi umum untuk metode berbasis matriks
