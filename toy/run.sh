#!/bin/bash
set -euo pipefail

echo "Running on $(hostname)"
echo "Start time: $(date)"

# Limita i thread ai core richiesti
export NUMBA_NUM_THREADS=64
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64

# Ambiente Python
source $PWD/venv/bin/activate

python -V
which python

# LANCIA IL TUO DRIVER
# Esegui il notebook con papermill
papermill \
  Run-Simulation.ipynb \
  Run-Simulation_out.ipynb

echo "End time: $(date)"
