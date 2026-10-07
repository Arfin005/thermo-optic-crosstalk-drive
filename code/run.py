"""One reconfiguration case: all strategies, evaluated on the reference chip."""
import sys, json, time, math
import numpy as np, torch
from drive import *
from ident import fit, step_responses

def evaluate(chip, V, PA_eval, Uss):
    S = chip.run(V, PA_eval)
    e = chip.opt.err(torch.tensor(S), Uss).numpy()
    t = TGRID
    return e, {'ts_1e-2': settling(t, e, 1e-2), 'ts_1e-3': settling(t, e, 1e-3),
               'iae_us': float(np.trapezoid(e, t) * 1e6), 'e_at_20us': float(np.interp(20e-6, t, e)),
               'e_at_100us': float(np.interp(100e-6, t, e))}

def case(seed, chip, dm, iters=400, verbose=False):
    rng = np.random.default_rng(seed)
    M = chip.fd.M; H = chip.H
    tA, tB = random_targets(rng, M), random_targets(rng, M)
    PA, PB = chip.static_powers(tA), chip.static_powers(tB)
    VB = chip.static_volts(PB)
    Uss = chip.opt.unitary(torch.tensor(chip.Sg @ PB))
    res = {'seed': seed, 'PB_max_mW': float(PB.max()*1e3), 'VB_max': float(VB.max()),
           'total_power_A_mW': float(PA.sum()*1e3), 'total_power_B_mW': float(PB.sum()*1e3)}
    curves = {}
    T = len(TGRID) - 1
    # naive
    PAn, PBn = chip.static_powers(tA, False), chip.static_powers(tB, False)
    e, r = evaluate(chip, np.repeat(chip.static_volts(PBn)[None], T, 0), PAn, Uss)
    curves['naive'] = e; res['naive'] = r; res['naive']['final_err'] = float(e[-1])
    # static
    e, r = evaluate(chip, np.repeat(VB[None], T, 0), PA, Uss); curves['static'] = e; res['static'] = r
    # siso
    Vs, ton = design_siso(dm, chip.opt, PA, PB, VB, chip.Hg @ PB)
    e, r = evaluate(chip, Vs, PA, Uss); curves['siso'] = e; res['siso'] = r
    res['siso']['mean_ton_us'] = float(ton.mean()*1e6)
    # mimo: design on model (its own static-error prediction as reference curve)
    t0 = time.time()
    with torch.no_grad():
        e_model_static = chip.opt.err(dm.run(torch.tensor(np.repeat(VB[None], T, 0)), PA), Uss).numpy()
    Vm, hist = design_mimo(dm, PA, VB, Uss, chip.opt, e_model_static, iters=iters, verbose=verbose)
    res['design_time_s'] = time.time() - t0
    e, r = evaluate(chip, Vm, PA, Uss); curves['mimo'] = e; res['mimo'] = r
    res['loss_hist'] = hist[::10]
    return res, curves, Vm, (PA, PB, VB)

if __name__ == '__main__':
    seed = int(sys.argv[1]); iters = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    tag = sys.argv[3] if len(sys.argv) > 3 else 'S0'
    chip = Chip()
    Y = np.load('ref_steps.npy'); dc = np.load('ref_dc.npy')
    R = fit(Y, TGRID, TAUS, dc=dc)
    dm = DesignModel(R, TAUS, chip.nseg, chip.el)
    res, curves, Vm, ctx = case(seed, chip, dm, iters=iters, verbose=True)
    print(json.dumps({k: v for k, v in res.items() if k != 'loss_hist'}, indent=1))
    json.dump(res, open(f'res_{tag}_{seed}.json', 'w'), indent=1)
    np.savez(f'cur_{tag}_{seed}.npz', t=TGRID, Vm=Vm, **curves)
