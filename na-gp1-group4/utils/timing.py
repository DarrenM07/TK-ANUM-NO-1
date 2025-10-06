import time                            # Library untuk mengukur waktu dengan presisi tinggi
from contextlib import contextmanager   # Digunakan untuk membuat context manager kustom (dengan 'with' statement)

@contextmanager
def timer():
    """
    Context manager sederhana untuk mengukur lama waktu eksekusi sebuah blok kode.

    Cara pakai:
        with timer() as t:
            # ... blok kode yang ingin diukur ...
        print(f"Waktu eksekusi: {t():.6f} detik")

    Penjelasan:
    - Fungsi ini memanfaatkan fitur 'contextmanager' dari modul contextlib.
    - Saat blok 'with' dimulai, waktu awal dicatat.
    - Saat blok selesai, fungsi lambda yang mengembalikan selisih waktu dikembalikan.
    """
    start = time.perf_counter()         # Catat waktu mulai (presisi tinggi, cocok untuk benchmark)
    yield lambda: time.perf_counter() - start  
    # 'yield' mengembalikan fungsi lambda yang jika dipanggil akan menghitung selisih waktu saat ini - waktu awal.
    # Jadi, saat blok 'with' selesai, kita bisa memanggil t() untuk mengetahui berapa lama waktu yang berlalu.
