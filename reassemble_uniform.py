import glob
import pickle

import numpy as np

# Stream the uniform-grid chunks into one compact float16 matrix (26.2M x 24),
# same format as full_results_arousal_ff_fb_isn.pkl ({"columns", "data"}).
directory = "/shome/ge74coy/pv_to_sst_code/results/"
saving = "/home/ge74coy/mnt/nasgroup/labmembers/fabioveneto/pv_to_sst_project/results/full_results_arousal_ff_fb_isn_uniform.pkl"

param_order = ["wEE", "wEP", "wES", "wPE", "wPP", "wPS",
               "wSE", "wSP", "wSV", "wVE", "wVP", "wVS"]
value_keys = [f"{k}_{cond}" for cond in ("low", "high")
              for k in ("lam", "L_EE", "L_EP", "L_EV", "gain_FF", "gain_FB")]
columns = param_order + value_keys

files = sorted(glob.glob(f"{directory}/results_arousal_isn_uniform_chunk_*.pkl"))
print(f"{len(files)} chunks")
assert len(files) == 1000

n_total = 0
for f in files:
    with open(f, "rb") as h:
        n_total += len(pickle.load(h))
assert n_total == 26_214_400, n_total
data = np.empty((n_total, len(columns)), dtype=np.float16)

row = 0
for f in files:
    with open(f, "rb") as h:
        chunk = pickle.load(h)
    for rec in chunk:
        data[row, :12] = rec["params"]
        for j, k in enumerate(value_keys):
            data[row, 12 + j] = rec[k]
        row += 1
assert row == n_total

print("wSP values:", sorted(set(np.unique(data[:1000000, 7]).tolist())))
print("wEP values:", sorted(set(np.unique(data[:1000000, 1]).tolist())))
with open(saving, "wb") as h:
    pickle.dump({"columns": columns, "data": data}, h, protocol=pickle.HIGHEST_PROTOCOL)
print(f"wrote {n_total:,} rows x {len(columns)} cols ({data.nbytes / 1e9:.2f} GB float16) -> {saving}")
