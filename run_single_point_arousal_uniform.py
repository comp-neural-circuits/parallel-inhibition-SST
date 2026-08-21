import sys
import pickle
import numpy as np
import sympy as sym

# Arousal scan, ISN UNIFORM grid (Gupta et al. neuromodulation model): same
# computation as run_single_point_arousal.py, but on the uniform-values grid —
# every weight uses {0.1, 0.7, 1.2, 1.8} (wEE: {1.1, 1.3, 1.5, 1.8}, ISN), and
# wSP/wSV additionally include 0 for exact ablation partners. For each weight
# configuration, evaluate the E gains to FF (E+PV) and FB (VIP) drive and the
# leading eigenvalue at LOW (theta=0) and HIGH (theta=1) arousal.
#
# PV->SST carries TM depression with release probability U_SP (lowered by ACh);
# at the fixed rate set-point it enters the linear theory as a scalar on wSP:
#   cSP = U_SP / (1 + U_SP*rP0*tau_SP)**2      (= pSP + DpSP*rP0)
# VIP->SST is static with gain 1 at both arousal levels (NE doubles the first
# pulse but halves the PPR -> no steady-state effect; not modeled), so only
# cSP differs between the conditions.
# Constants below must match the VIP_circuit_neuromod defaults in util.py.

def index_to_point(idx, grids):
    # grids = list of 1D arrays
    dims = [len(g) for g in grids]
    point = []
    for d in reversed(dims):
        idx, pos = divmod(idx, d)
        point.append(pos)
    return [grids[i][p] for i, p in enumerate(reversed(point))]


job_id = int(sys.argv[1])
n_jobs = int(sys.argv[2])

with open("grid_meta_wsp_wsv_isn_uniform.pkl","rb") as f:
    meta = pickle.load(f)

N = meta["N"]
grids = meta["param_grids"]

chunk_size = N // n_jobs
start = job_id * chunk_size
end = (job_id+1)*chunk_size if job_id < n_jobs-1 else N

# ==== symbolic objects ====

bE, bP, bS, bV = sym.symbols('bE bP bS bV', positive=True)
wEE, wEP, wES, wPE, wPP, wPS, wSE, wSP, wSV, wVE, wVP, wVS = sym.symbols(
    'wEE wEP wES wPE wPP wPS wSE wSP wSV wVE wVP wVS', real=True
)
cSP, cSV = sym.symbols('cSP cSV', positive=True)
weights = [wEE, wEP, wES, wPE, wPP, wPS, wSE, wSP, wSV, wVE, wVP, wVS]

D = sym.diag(1/bE, 1/bP, 1/bS, 1/bV)
W = sym.Matrix([
    [ wEE, -wEP, -wES,  0],
    [ wPE, -wPP, -wPS,  0],
    [ wSE, -cSP*wSP,   0,  -cSV*wSV],
    [ wVE, -wVP, -wVS,   0]
])
J = W - D      # stability
A = D - W      # linear response

detA = sym.simplify(A.det())
adjA = sym.simplify(A.adjugate())

L_EE = adjA[0, 0] / detA
L_EP = adjA[0, 1] / detA
L_EV = adjA[0, 3] / detA

# fixed values (exact for power=1, mult_f=1); cSV = 1 (VIP->SST gain fixed)
fixed_vals = {bE: 1, bP: 1, bS: 1, bV: 1, cSV: 1}

args_syms = weights + [cSP]

J_func    = sym.lambdify(args_syms, J.subs(fixed_vals), 'numpy')
det_func  = sym.lambdify(args_syms, detA.subs(fixed_vals), 'numpy')
L_EE_func = sym.lambdify(args_syms, L_EE.subs(fixed_vals), 'numpy')
L_EP_func = sym.lambdify(args_syms, L_EP.subs(fixed_vals), 'numpy')
L_EV_func = sym.lambdify(args_syms, L_EV.subs(fixed_vals), 'numpy')

# ==== arousal conditions (must match VIP_circuit_neuromod defaults) ====

tau_SP = 0.100
U_SP_base, U_SP_high = 0.5, 0.15
rP0 = 5.0
I_FF_E = 1.0
I_FF_P = 1.0
I_FB_V = 1.0

def c_SP_of(U):
    return U / (1.0 + U * rP0 * tau_SP)**2

conditions = [("low",  c_SP_of(U_SP_base)),
              ("high", c_SP_of(U_SP_high))]

# --- run chunk ---
results = []

for lin_idx in range(start, end):
    point = index_to_point(lin_idx, grids)
    rec = {"params": np.array(point, dtype=np.float16)}

    for cond, cSP_val in conditions:
        vals = list(point) + [cSP_val]

        det_val = float(det_func(*vals))
        if abs(det_val) > 1e-6:
            L_EE_val = float(L_EE_func(*vals))
            L_EP_val = float(L_EP_func(*vals))
            L_EV_val = float(L_EV_func(*vals))
            gain_ff = I_FF_E * L_EE_val + I_FF_P * L_EP_val
            gain_fb = I_FB_V * L_EV_val
        else:
            L_EE_val = L_EP_val = L_EV_val = np.nan
            gain_ff = gain_fb = np.nan

        J_num = np.asarray(J_func(*vals), dtype=float)
        lam = np.max(np.real(np.linalg.eigvals(J_num)))

        rec[f"lam_{cond}"] = np.float16(lam)
        rec[f"L_EE_{cond}"] = np.float16(L_EE_val)
        rec[f"L_EP_{cond}"] = np.float16(L_EP_val)
        rec[f"L_EV_{cond}"] = np.float16(L_EV_val)
        rec[f"gain_FF_{cond}"] = np.float16(gain_ff)
        rec[f"gain_FB_{cond}"] = np.float16(gain_fb)

    results.append(rec)

# save chunk
outfile = f"/shome/ge74coy/pv_to_sst_code/results/results_arousal_isn_uniform_chunk_{job_id:04d}.pkl"
with open(outfile, "wb") as f:
    pickle.dump(results, f)
