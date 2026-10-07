"""Single reconfigurations on the 16x16 mesh (240 heaters): naive, static, per-heater
pre-emphasis and proposed, all evaluated on the 3-D modal model.  Args: seed"""
import os, sys, json, time
import numpy as np, torch
torch.set_num_threads(1)
from drive import *
from fd_thermal import Layout
from qp import basis
from run import evaluate
os.makedirs('out', exist_ok=True)

seed = int(sys.argv[1])
chip = Chip(lay=Layout(N=16))
R = np.load('R16.npy')
dm = DesignModel(R, TAUS, chip.nseg, chip.el)
rng = np.random.default_rng(seed)
M = chip.fd.M; H = chip.H; T = len(TGRID) - 1
tA, tB = random_targets(rng, M), random_targets(rng, M)
PA, PB = chip.static_powers(tA), chip.static_powers(tB)
VA, VB = chip.static_volts(PA), chip.static_volts(PB)
Uss = chip.opt.unitary(torch.tensor(chip.Sg @ PB))
res = {'seed': seed, 'N': 16, 'H': H, 'PB_max_mW': float(PB.max()*1e3), 'VB_max': float(VB.max()),
       'total_power_A_mW': float(PA.sum()*1e3), 'total_power_B_mW': float(PB.sum()*1e3)}
cur = {}
PAn, PBn = chip.static_powers(tA, False), chip.static_powers(tB, False)
e, r = evaluate(chip, np.repeat(chip.static_volts(PBn)[None], T, 0), PAn, Uss); cur['naive'] = e; res['naive'] = r; r['final_err'] = float(e[-1])
Vst = np.repeat(VB[None], T, 0)
e, r = evaluate(chip, Vst, PA, Uss); cur['static'] = e; res['static'] = r
Vs, ton = design_siso(dm, chip.opt, PA, PB, VB, chip.Hg @ PB)
e, r = evaluate(chip, Vs, PA, Uss); cur['siso'] = e; res['siso'] = r
with torch.no_grad():
    e_ms = chip.opt.err(dm.run(torch.tensor(Vst), PA), Uss).numpy()
t1 = time.time()
Vl, zl, hist = design_lbfgs(dm, PA, VB, Uss, chip.opt, e_ms, lambda tk: basis(tk, taus=TAUS),
                            outer=5, delta=1e-4, Vbase=Vs)
res['design_s'] = time.time() - t1; res['n_evals'] = len(hist); res['nvar'] = int(zl.size)
e, r = evaluate(chip, Vl, PA, Uss); cur['prop'] = e; res['prop'] = r
with torch.no_grad():
    em = chip.opt.err(dm.run(torch.tensor(Vl), PA), Uss).numpy()
res['prop_model'] = {'ts_1e-2': settling(TGRID, em, 1e-2), 'ts_1e-3': settling(TGRID, em, 1e-3)}
res['Vl_max'] = float(Vl.max()); res['Vl_min'] = float(Vl.min())
json.dump(res, open(f'out/res_N16_{seed}.json', 'w'), indent=1)
np.savez(f'out/cur_N16_{seed}.npz', t=TGRID, Vs=Vs, Vl=Vl, **cur)
print('N16', seed, {k: {a: round(b*1e6, 1) if a.startswith('ts') else round(b, 4) for a, b in res[k].items()} for k in ('static', 'siso', 'prop')}, 'design %.0fs' % res['design_s'], flush=True)
