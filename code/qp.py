"""Convex (QP) design of crosstalk-cancelling heater drive waveforms.

Heater powers  P_h(t) = P_B,h + sum_b theta_{b,h} g_b(t)  with basis functions g_b that vanish
as t grows (piecewise-linear hats + decaying exponentials at the identified thermal poles), so
every design converges to the static calibration P_B.
With the identified shared-pole model the segment deviations are affine in theta:
    dy(t) = d0(t) + A(t) theta,
and near the target the (row-phase-aligned) mesh error is quadratic in the phase deviations:
    e(t)^2 ~= || L beta dy(t) ||^2,   L^T L = J^T J / N,  J = d(aligned U)/d(phases) at target.
Minimising a time-weighted sum of these errors subject to 0 <= P_h(t) <= P_max,h(t) (driver
voltage limit) is a convex quadratic programme.
"""
import math, time
import numpy as np
import torch
import scipy.sparse as sp
import cvxpy as cp
from ident import TGRID


def metric_factor(opt, seg_ss, Uss, mu=0.0):
    N = opt.N
    seg0 = torch.tensor(seg_ss)

    def f(dphi):
        U = opt.unitary(seg0 + dphi / opt.beta)
        inner = (Uss.conj() * U).sum(-1)
        U = U * (inner.conj() / inner.abs())[..., None]
        D = (U - Uss) / math.sqrt(N)
        return torch.cat([D.real.reshape(-1), D.imag.reshape(-1)])
    J = torch.autograd.functional.jacobian(f, torch.zeros(len(seg_ss))).numpy()   # (2N^2, nseg)
    Mm = J.T @ J
    Mm = Mm + mu * np.trace(Mm) / Mm.shape[0] * np.eye(Mm.shape[0])   # penalise all phase deviations
    w, V = np.linalg.eigh(Mm)
    keep = w > 1e-10 * w.max()
    return (np.sqrt(w[keep])[:, None] * V[:, keep].T)                            # (r, nseg)


def basis(tk, Te_fine=40e-6, taus=None):
    """Hat functions on knots (1 us to 40 us, 10 us to 200 us, 100 us to 1 ms; zero at 1 ms)
    plus exponentials exp(-t/tau) for identified poles tau >= 2 us.  Returns (B, len(tk))."""
    knots = np.concatenate([np.arange(0, 40e-6, 1e-6), np.arange(40e-6, 200e-6, 10e-6),
                            np.arange(200e-6, 1e-3 + 1e-12, 100e-6)])
    G = []
    for i in range(len(knots) - 1):
        left = knots[i - 1] if i > 0 else None
        c, right = knots[i], knots[i + 1]
        g = np.zeros_like(tk)
        m = (tk >= c) & (tk < right)
        g[m] = (right - tk[m]) / (right - c)
        if left is not None:
            m = (tk >= left) & (tk < c)
            g[m] = (tk[m] - left) / (c - left)
        G.append(g)
    for tau in taus[taus >= 2e-6]:
        G.append(np.exp(-tk / tau))
    return np.array(G)


def filtered(G, taus):
    """ZOH first-order responses z[b,k,n] (zero initial state) on TGRID; n = 0..T."""
    dt = np.diff(TGRID)
    d = np.exp(-dt[:, None] / taus[None, :])                    # (T, K)
    B, T = G.shape
    z = np.zeros((B, len(taus), T + 1))
    for n in range(T):
        z[:, :, n + 1] = d[n][None, :] * z[:, :, n] + (1 - d[n])[None, :] * G[:, n][:, None]
    return z


def design_qp(dm_R, taus, nseg, el, opt, PA, PB, Uss, e_static, delta=1e-4, ridge=1e-9,
              n_obj=260, iters=2, verbose=False, mu=0.05):
    """Returns voltage waveform V (T, H) on the ZOH grid and diagnostics."""
    t0 = time.time()
    R = dm_R                                                     # (O+H, H, K)
    H, K = R.shape[1], R.shape[2]
    tk = TGRID[:-1]
    G = basis(tk, taus=taus)                                     # (B, T)
    B = G.shape[0]
    z = filtered(G, taus)                                        # (B, K, T+1)
    L = metric_factor(opt, (R[:nseg].sum(-1) @ PB), Uss, mu) * opt.beta   # (r, nseg), per K m
    LR = np.einsum('rs,shk->rhk', L, R[:nseg])                   # (r, H, K)
    # objective sample indices, log-spaced in time (n >= 1)
    idx = np.unique(np.round(np.interp(np.logspace(np.log10(TGRID[1]), np.log10(TGRID[-1]), n_obj),
                                       TGRID, np.arange(len(TGRID)))).astype(int))
    idx = idx[idx >= 1]
    tt = TGRID[idx]
    lw = np.gradient(np.log(tt))
    wts = lw / (e_static[idx] ** 2 + delta ** 2)
    wts = wts / wts.max()
    dP0 = (PA - PB)                                              # W
    nv = B * H
    Hm = np.zeros((nv, nv)); qv = np.zeros(nv); c0 = 0.0
    for s in range(0, len(idx), 20):
        ii = idx[s:s + 20]
        F = np.einsum('rhk,bkt->trbh', LR, z[:, :, ii]).reshape(len(ii), L.shape[0], nv) * 1e-3  # per mW
        d0 = np.einsum('rhk,h,tk->tr', LR, dP0, np.exp(-TGRID[ii][:, None] / taus[None, :]))
        w = wts[s:s + 20]
        Hm += np.einsum('t,tri,trj->ij', w, F, F, optimize=True)
        qv += np.einsum('t,tri,tr->i', w, F, d0)
        c0 += (w * (d0 ** 2).sum(1)).sum()
    Hm = 0.5 * (Hm + Hm.T) + ridge * np.trace(Hm) / nv * np.eye(nv)
    if verbose:
        print(f'   QP built: {nv} variables, rank {L.shape[0]}, {time.time()-t0:.0f}s', flush=True)
    # constraints on a grid up to 1.2 ms (bases vanish later except slow exponentials, which
    # are also checked at a few later times)
    cidx = np.nonzero(tk <= 1.2e-3)[0]
    cidx = np.unique(np.concatenate([cidx, np.searchsorted(tk, [2e-3, 4e-3, 8e-3])]))
    Gc = G[:, cidx]                                              # (B, nc)
    rows, cols, vals = [], [], []
    for h in range(H):
        for jb in range(B):
            nzr = np.nonzero(Gc[jb])[0]
            rows.append(h * len(cidx) + nzr); cols.append(np.full(len(nzr), jb * H + h)); vals.append(Gc[jb, nzr])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(H * len(cidx), nv))
    PBmW = PB * 1e3
    # heater temperatures for the voltage limit: start from the static-step trajectory
    Th = None
    theta_v = np.zeros(nv)
    for it in range(iters):
        if Th is None:
            Th_c = np.einsum('ohk,hk->o', R[nseg:], np.repeat(PB[:, None], K, 1))[:, None] * np.ones((1, len(cidx)))
            Th_c = np.maximum(Th_c, np.einsum('ohk,hk->o', R[nseg:], np.repeat(PA[:, None], K, 1))[:, None])
        else:
            Th_c = Th[:, cidx]
        Rh = el.R0 * (1 + el.alpha * Th_c)
        Pmax = el.Vmax ** 2 * Rh / (Rh + el.Rs) ** 2 * 1e3 * 0.995          # mW, (H, nc)
        ub = (Pmax - PBmW[:, None]).reshape(-1)
        lb = (-PBmW[:, None] * np.ones((1, len(cidx)))).reshape(-1)
        th = cp.Variable(nv)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(th, cp.psd_wrap(Hm)) + qv @ th),
                          [A @ th <= ub, A @ th >= lb])
        prob.solve(solver=cp.CLARABEL)
        theta_v = th.value
        # model heater temperatures along the designed trajectory (for the next Pmax update)
        Pt = PB[:, None] + (theta_v.reshape(B, H).T @ G) * 1e-3               # (H, T) W
        Th = model_heater_temps(R, taus, nseg, PA, Pt)
        if verbose:
            print(f'   QP iter {it}: status {prob.status}, obj {prob.value + 0.5*c0:.4e} '
                  f'(static {0.5*c0:.4e}), {time.time()-t0:.0f}s', flush=True)
    Pt = PB[:, None] + (theta_v.reshape(B, H).T @ G) * 1e-3
    Pt = np.clip(Pt, 0, None)
    V = el.volt(Pt, Th[:, :-1]).T                                   # (T, H)
    return np.clip(V, 0, el.Vmax), {'obj': float(prob.value + 0.5 * c0), 'obj_static': float(0.5 * c0),
                                     'build_solve_s': time.time() - t0, 'nvar': nv}


def model_heater_temps(R, taus, nseg, PA, Pt):
    """Heater temperatures (H, T+1) predicted by the design model for power trajectory Pt (H,T)."""
    dt = np.diff(TGRID)
    x = np.repeat(PA[:, None], len(taus), 1)
    out = [np.einsum('ohk,hk->o', R[nseg:], x)]
    for n in range(Pt.shape[1]):
        d = np.exp(-dt[n] / taus)
        x = d[None, :] * x + (1 - d)[None, :] * Pt[:, n][:, None]
        out.append(np.einsum('ohk,hk->o', R[nseg:], x))
    return np.array(out).T
