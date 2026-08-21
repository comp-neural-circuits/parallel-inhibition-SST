#!/bin/bash
cd /shome/ge74coy/pv_to_sst_code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/shome/ge74coy/.conda/envs/linear_theory/bin/python
seq 0 999 | xargs -P 200 -I{} $PY run_single_point_arousal_uniform.py {} 1000
echo "workers done, chunks: $(ls /shome/ge74coy/pv_to_sst_code/results/results_arousal_isn_uniform_chunk_*.pkl 2>/dev/null | wc -l)"
