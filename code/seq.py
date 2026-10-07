"""Sequences of reconfigurations with a short dwell time (tiled / time-multiplexed matrix
operations).  The mesh is reprogrammed every Td; the next reconfiguration starts before the
substrate heat of the previous ones has settled.

Strategies, per window s (target configuration s, previous configuration s-1):
  static    : voltage step to the static calibration of s
  siso      : per-heater pre-emphasis (designed as if starting from steady state of s-1)
  prop_ss   : proposed design computed for a start from the steady state of s-1 (as in the
              single-reconfiguration study), applied unchanged in the sequence
  prop_hist : proposed design computed from the compact model's predicted thermal state at the
              end of window s-1 (history-aware feedforward)
All strategies are evaluated together on the 3-D modal model, over the whole sequence.
Args: seed Td_us S
"""
import os, sys, json, time
import numpy as np, torch
torch.set_num_threads(1)
from drive import *
from ident import fit
from qp import basis
from fd_thermal import time_grid
os.makedirs('out', exist_ok=True)

seed, Td, S = int(sys.argv[1]), float(sys.argv[2]) * 1e-6, int(sys.argv[3])
tw = time_grid(((100e-6, 0.25e-6), (300e-6, 1e-6), (1e-3, 5e-6), (10e-3, 20e-6)))
tw = tw[tw <= Td + 1e-12]
Tn = len(tw) - 1
chip = Chip()
Y = np.load('ref_steps.npy'); dc = np.load('ref_dc.npy')
R = fit(Y, TGRID, TAUS, dc=dc)
dm = DesignModel(R, TAUS, chip.nseg, chip.el)
R2 = dm.R2; taus_t = dm.taus
dec = torch.exp(-torch.tensor(np.diff(tw))[:, None] / taus_t[None, :])
el = chip.el; Vmax = el.Vmax; H = chip.H; nseg = chip.nseg


def mrun(V, x0):
    """Compact model on the window grid from state x0 (H, K); returns segs (Tn+1, nseg), x_end."""
    x = x0.clone()
    y = R2 @ x.reshape(-1)
    segs = [y[:nseg]]
    for k in range(V.shape[0]):
        Th = y[nseg:]
        Rh = el.R0 * (1 + el.alpha * Th)
        P = V[k] ** 2 * Rh / (Rh + el.Rs) ** 2
        d = dec[k]
        x = d[None, :] * x + (1 - d)[None, :] * P[:, None]
        y = R2 @ x.reshape(-1)
        segs.append(y[:nseg])
    return torch.stack(segs), x


G = basis(tw[:-1], taus=TAUS)
G = torch.tensor(G[np.abs(G).max(1) > 0])
lw = torch.tensor(np.gradient(np.log(tw + 1e-7)))


def design(x0, VB, Vbase, Uss, outer=5, inner=20, delta=1e-4):
    with torch.no_grad():
        eref = chip.opt.err(mrun(torch.tensor(np.repeat(VB[None], Tn, 0)), x0)[0], Uss)
    w = lw / (eref ** 2 + delta ** 2); w = w / w.sum()
    zB = torch.logit(torch.tensor(np.clip(Vbase / Vmax, 0.01, 0.99)))
    tail = torch.tensor(tw[:-1] > 150e-6)[:, None]
    zB = torch.where(tail, torch.logit(torch.tensor(VB) / Vmax)[None, :], zB)
    z = torch.zeros(G.shape[0], H, requires_grad=True)
    wave = lambda z: Vmax * torch.sigmoid(zB + G.T @ z)
    opt_ = torch.optim.LBFGS([z], lr=1.0, max_iter=inner, history_size=30, line_search_fn='strong_wolfe',
                             tolerance_grad=1e-12, tolerance_change=1e-14)
    n = [0]
    def closure():
        opt_.zero_grad()
        L = (w * chip.opt.err(mrun(wave(z), x0)[0], Uss) ** 2).sum()
        L.backward(); n[0] += 1
        return L
    for _ in range(outer):
        opt_.step(closure)
    with torch.no_grad():
        return wave(z).numpy(), n[0]


def siso_wave(PA, PB, VB):
    _, ton = design_siso(dm, chip.opt, PA, PB, VB, chip.Hg @ PB)
    V = np.repeat(VB[None], Tn, 0)
    up = PB > PA
    for h in range(H):
        V[tw[:-1] < ton[h], h] = Vmax if up[h] else 0.0
    return V


rng = np.random.default_rng(1000 + seed)
th = [random_targets(rng, chip.fd.M) for _ in range(S + 1)]
P = [chip.static_powers(t) for t in th]
VB = [chip.static_volts(p) for p in P]
Uss = [chip.opt.unitary(torch.tensor(chip.Sg @ p)) for p in P]
xss = lambda p: torch.tensor(p)[:, None].expand(-1, len(TAUS)).clone()

waves = {k: [] for k in ('static', 'siso', 'prop_ss', 'prop_hist')}
x_hist = xss(P[0])
t0 = time.time(); nev = 0
for s in range(1, S + 1):
    Vs = siso_wave(P[s - 1], P[s], VB[s])
    waves['static'].append(np.repeat(VB[s][None], Tn, 0))
    waves['siso'].append(Vs)
    Vss, n1 = design(xss(P[s - 1]), VB[s], Vs, Uss[s])
    waves['prop_ss'].append(Vss)
    Vh, n2 = design(x_hist, VB[s], Vs, Uss[s])
    waves['prop_hist'].append(Vh)
    with torch.no_grad():
        _, x_hist = mrun(torch.tensor(Vh), x_hist)
    nev += n1 + n2
    print(f'window {s} designed ({time.time()-t0:.0f} s)', flush=True)
design_s = time.time() - t0

# evaluate on the 3-D model over the whole sequence
tg = np.concatenate([[0.0]] + [s * Td + tw[1:] for s in range(S)])
res = {'seed': seed, 'Td_us': Td * 1e6, 'S': S, 'design_s': design_s, 'n_evals': nev}
cur = {'t': tg}
for k, W in waves.items():
    V = np.concatenate(W, 0)
    c0 = chip.md.steady_state_modes(P[0])
    Sg_, _, _ = chip.md.simulate(tg, lambda i, Th: el.power(V[i], Th), c0)
    errs, ts2, ts3, eend = [], [], [], []
    for s in range(S):
        seg = Sg_[s * Tn: (s + 1) * Tn + 1]
        e = chip.opt.err(torch.tensor(seg), Uss[s + 1]).numpy()
        errs.append(e)
        ts2.append(settling(tw, e, 1e-2)); ts3.append(settling(tw, e, 1e-3)); eend.append(float(e[-1]))
    cur[k] = np.concatenate(errs)
    res[k] = {'ts_1e-2': ts2, 'ts_1e-3': ts3, 'e_end': eend}
    print(k, 'ts1e-2', [round(x * 1e6, 1) for x in ts2], 'ts1e-3', [round(x * 1e6, 1) for x in ts3], flush=True)
tag = f'seq_{int(Td*1e6)}_{seed}'
json.dump(res, open(f'out/res_{tag}.json', 'w'), indent=1)
np.savez(f'out/cur_{tag}.npz', tw=tw, **cur, **{f'V_{k}': np.concatenate(W, 0) for k, W in waves.items()})
