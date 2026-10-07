"""DAC resolution: drive waveforms quantised to n bits over [0, Vmax]; error measured against the
steady state the quantised static voltages settle to (so only the dynamics are compared)."""
import json, sys
import numpy as np, torch
torch.set_num_threads(1)
from drive import *
from run import evaluate
chip = Chip(); T = len(TGRID) - 1; Vmax = chip.el.Vmax
out = []
for s in [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]:
    d = np.load(f'out/cur_main_{s}.npz')
    PA = d['PA']
    for bits in (10, 12, 14, 16):
        q = lambda V: np.round(V / Vmax * (2 ** bits - 1)) / (2 ** bits - 1) * Vmax
        VBq = q(d['VB'])
        P = chip.static_powers(d['tB'])
        for _ in range(50):
            P = chip.el.power(VBq, chip.Hg @ P)
        Ussq = chip.opt.unitary(torch.tensor(chip.Sg @ P))
        Uss = chip.opt.unitary(torch.tensor(chip.Sg @ chip.static_powers(d['tB'])))
        floor = float(chip.opt.err(torch.tensor(chip.Sg @ P)[None], Uss)[0])
        row = {'seed': s, 'bits': bits, 'static_floor': floor}
        for k, V in (('static', np.repeat(VBq[None], T, 0)), ('prop', q(d['Vl']))):
            e, r = evaluate(chip, V, PA, Ussq); row[k] = r
        out.append(row)
        print(s, bits, 'floor %.1e' % floor, {k: (round(row[k]['ts_1e-2']*1e6, 1), round(row[k]['ts_1e-3']*1e6, 1)) for k in ('static', 'prop')}, flush=True)
json.dump(out, open('out/quant.json', 'w'), indent=1)
