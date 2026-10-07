"""Fabrication variations: waveforms designed on the nominal model/chip are applied, as an additive
correction to each chip's own static calibration, to chips with different stacks."""
import sys, json
import numpy as np, torch
torch.set_num_threads(1)
from drive import *
from fd_thermal import Stack
from run import evaluate

corners = {
    'k_ox -10%': dict(scale={'k_ox': 0.9}), 'k_ox +10%': dict(scale={'k_ox': 1.1}),
    't_box 1.8um': dict(stack=Stack(t_box=1.8e-6)), 't_box 2.2um': dict(stack=Stack(t_box=2.2e-6)),
    'z_h 1.8um': dict(stack=Stack(z_h=1.8e-6)), 'z_h 2.2um': dict(stack=Stack(z_h=2.2e-6)),
    'R0 -5%': dict(elec=Elec(R0=475.0)), 'R0 +5%': dict(elec=Elec(R0=525.0)),
}
seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
out = []
for name, kw in corners.items():
    chip = Chip(**kw)
    for s in seeds:
        d = np.load(f'out/cur_main_{s}.npz')
        tA, tB = d['tA'], d['tB']
        PA, PB = chip.static_powers(tA), chip.static_powers(tB)
        VB = chip.static_volts(PB)
        Uss = chip.opt.unitary(torch.tensor(chip.Sg @ PB))
        T = len(TGRID) - 1
        row = {'corner': name, 'seed': s}
        for k, V in (('static', np.repeat(VB[None], T, 0)),
                     ('siso', np.clip(VB[None] + d['Vs'] - d['VB'][None], 0, chip.el.Vmax)),
                     ('prop', np.clip(VB[None] + d['Vl'] - d['VB'][None], 0, chip.el.Vmax))):
            e, r = evaluate(chip, V, PA, Uss)
            row[k] = r
        out.append(row)
        print(name, s, {k: (round(row[k]['ts_1e-2']*1e6, 1), round(row[k]['ts_1e-3']*1e6, 1)) for k in ('static', 'siso', 'prop')}, flush=True)
json.dump(out, open('out/variation.json', 'w'), indent=1)
