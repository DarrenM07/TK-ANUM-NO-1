# TK-ANUM-NO-1
K4, TK ANUM NO 1, (Darren Marcello Sidabutar, Hafizh Surya Mustafa Zen, Chiara Aqmarina Diankusumo, Ameera Khaira Tawfiqa)

# HOW TO RUN NUMBER 1

# 1. Requirements
Python 3.11+ (tested)

Packages: numpy, pandas, matplotlib

# 2. Quick start (reproducible run)

Create a clean virtual environment and install deps:
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pandas

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate
pip install numpy matplotlib pandas

# Run the experiment:

python main_vandermonde.py

# This will generate:
- CSV: outputs/vandermonde/results_with_chebyshev.csv
    Columns: nodes,n,solver,time_ms,rel_err,cond1_est,flops_model,mem_model_bytes

- Figures:
    outputs/vandermonde/runtime_comparison.png
    outputs/vandermonde/error_comparison.png

