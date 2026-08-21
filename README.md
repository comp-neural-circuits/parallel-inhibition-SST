# Arousal control of the E–PV–SST–VIP microcircuit via PV→SST short-term depression

Model and analyses for the neuromodulation results of Gupta et al.: a four-population
rate model (E, PV, SST, VIP) in which the PV→SST synapse expresses Tsodyks–Markram
short-term depression whose release probability is the target of cholinergic arousal.
All results live in one notebook.

## Contents

| File | Role |
|---|---|
| `analysis_neuromod_ff_fb.ipynb` | All figures and grid analyses (the source of results) |
| `util.py` | Model classes; the notebook uses `VIP_circuit_neuromod` (linear theory, arousal sweep, simulations) |
| `grid_meta_wsp_wsv_isn_uniform.pkl` | Grid definition: axes and N = 26,214,400 |
| `run_single_point_arousal_uniform.py` | Grid worker: gains, response-matrix entries and leading eigenvalue at both arousal ends, per weight configuration |
| `run_single_chunk_arousal_uniform.sh` | SLURM array (0–999), same chunks |
| `run_uniform_grid.sh` | Same as SLURM but can be run on a single machine |
| `reassemble_uniform.py` | Streams the 1000 chunks into the compact pickle `full_results_arousal_ff_fb_isn_uniform.pkl` (26.2M × 24 float16, `{"columns", "data"}`) (~1.2 GB)|

Not tracked here because of the size (`full_results_arousal_ff_fb_isn_uniform.pkl`). The
notebook reads it via a `LOAD`/`GRID_PATH` switch defined in the first cell.

## Environments

- **Notebook (local)**: conda env — Python 3.11 with `numpy`, `scipy`,
  `matplotlib`, `seaborn`, `pandas`. (`util.py` needs `scipy.linalg`.)

## Getting the grid data

Default — the notebook's first cell sets

```python
LOAD = True
GRID_PATH = path + "/full_results_arousal_ff_fb_isn_uniform.pkl"
```

With `LOAD = False` the loaders raise an error reminding you that the simulations must be run first — regenerate the grid as below,
then point `GRID_PATH` at the resulting file.

To regenerate from scratch:

```bash
# 1. stage inputs on the cluster
rsync -avz grid_meta_wsp_wsv_isn_uniform.pkl run_single_point_arousal_uniform.py \
      "server:directory"

# 2. run the workers on a compute node
run_uniform_grid.sh
# (or submit run_single_chunk_arousal_uniform.sh as a SLURM array where a queue is available)

# 3. reassemble into the compact pickle in the notebook's default GRID_PATH
reassemble_uniform.py
```

Paths inside `run_uniform_grid.sh` and `reassemble_uniform.py` are my personal nas
locations. The grid axes: every weight takes values {0.1, 0.7, 1.2, 1.8}, `wEE` ∈
{1.1, 1.3, 1.5, 1.8} (ISN by construction), and `wSP`, `wSV` additionally include 0 —
so every circuit has exact ablation partners in the grid.

## Running the notebook
The matrix cell defines `W_C` (+ variants) and ends with `W_RUN = W_C`. Every analysis cell runs on `W_RUN`; switch that one assignment and re-run to regenerate the notebook for another weight configuration. Grid-population statistics are independent of `W_RUN`.