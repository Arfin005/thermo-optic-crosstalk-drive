"""Drive design on the identified model; evaluation on the 3-D reference.

Strategies compared (all evaluated on the reference chip, including heater self-heating):
  naive   : per-heater calibration only (no crosstalk knowledge), single step
  static  : steady-state crosstalk compensation (matrix inverse), single step   [prior art]
  siso    : static compensation + per-heater bang-bang pre-emphasis designed on each heater's
            own identified response
  mimo    : proposed; waveforms for all heaters jointly optimised by gradient descent through
            the identified model and the mesh transfer matrix
"""
import math, json, sys, time
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from fd_thermal import FDMesh, Stack, Layout
from fd_modal import ModalFD
from ident import TGRID, fit, step_responses

TAUS = np.logspace(np.log10(0.25e-6), np.log10(2.4e-3), 24)


class Elec:
    def __init__(self, R0=500.0, Rs=50.0, alpha=1e-3, Vmax=8.0):
        self.R0, self.Rs, self.alpha, self.Vmax = R0, Rs, alpha, Vmax

    def power(self, V, Th):
        R = self.R0 * (1 + self.alpha * Th)
        return V ** 2 * R / (R + self.Rs) ** 2

    def volt(self, P, Th):
        R = self.R0 * (1 + self.alpha * Th)
        return (R + self.Rs) * np.sqrt(np.clip(P, 0, None) / R)


class Optics:
    """Clements mesh transfer matrix from segment phases; row-phase-aligned error metric."""
    def __init__(self, fd):
        self.N = fd.lay.N
        self.mzis = fd.mzis
        self.beta = fd.beta

    def unitary(self, seg):                  # seg: (..., 2N*N) integrals [K m] (torch)
        N = self.N
        ph = torch.exp(1j * self.beta * seg).reshape(*seg.shape[:-1], 2 * N, N)
        U = torch.eye(N, dtype=torch.complex128).expand(*seg.shape[:-1], N, N).clone()
        r = 1 / math.sqrt(2)
        for c in range(N):
            top = torch.tensor(list(range(c % 2, N - 1, 2)))
            bot = top + 1
            for s in (0, 1):
                U = ph[..., 2 * c + s, :, None] * U
                ut, ub = U[..., top, :], U[..., bot, :]
                U = U.clone()
                U[..., top, :] = r * (ut + 1j * ub)
                U[..., bot, :] = r * (1j * ut + ub)
        return U

    def err(self, seg, Uss):
        U = self.unitary(seg)
        inner = (Uss.conj() * U).sum(-1)
        U = U * (inner.conj() / (inner.abs() + 1e-30))[..., None]
        return torch.linalg.matrix_norm(U - Uss) / math.sqrt(self.N)

    def diff_phases(self, seg):              # numpy (…, nseg) -> (…, H)
        N, b = self.N, self.beta
        e = [b * (seg[..., (2 * c) * N + m] - seg[..., (2 * c) * N + m + 1]) for (c, m) in self.mzis]
        i = [b * (seg[..., (2 * c + 1) * N + m] - seg[..., (2 * c + 1) * N + m + 1]) for (c, m) in self.mzis]
        return np.stack(e + i, -1)


class Chip:
    """A reference chip: modal 3-D thermal model + electrical parameters + static calibration."""
    def __init__(self, stack=None, scale=None, elec=None, lay=None):
        self.fd = FDMesh(st=stack or Stack(), lay=lay or Layout())
        self.md = ModalFD(self.fd, scale)
        self.el = elec or Elec()
        Sg, Hg = self.md.dc_gains()
        self.Sg, self.Hg = Sg, Hg
        self.opt = Optics(self.fd)
        self.nseg, self.H = self.fd.nseg, self.fd.H
        self.K = self.opt.diff_phases(Sg.T).T          # (H diff phases, H heaters) rad/W

    def static_powers(self, target, compensate=True):
        if not compensate:
            return target / np.diag(self.K)
        tgt = target.copy()
        for _ in range(20):
            P = np.linalg.solve(self.K, tgt)
            if (P >= 0).all():
                return P
            tgt = tgt + 2 * math.pi * (P < 0)
        raise RuntimeError('no non-negative solution')

    def static_volts(self, P):
        return self.el.volt(P, self.Hg @ P)

    def run(self, V, P0):
        """Simulate voltage waveform V (T-1, H) from steady state at powers P0."""
        c0 = self.md.steady_state_modes(P0)
        el = self.el
        S, Hh, _ = self.md.simulate(TGRID, lambda k, Th: el.power(V[k], Th), c0)
        return S


class DesignModel(torch.nn.Module):
    """Identified shared-pole model in torch."""
    def __init__(self, R, taus, nseg, elec):
        super().__init__()
        self.R = torch.tensor(R)                      # (O+H, H, K)
        self.R2 = self.R.reshape(self.R.shape[0], -1).contiguous()   # (O+H, H*K) for fast mv
        self.taus = torch.tensor(taus)
        self.nseg = nseg
        self.el = elec
        dt = torch.tensor(np.diff(TGRID))
        self.dec = torch.exp(-dt[:, None] / self.taus[None, :])   # (T-1, K)

    def run(self, V, P0):
        x = torch.tensor(P0)[:, None].expand(-1, len(self.taus)).clone()   # steady at P0
        y = self.R2 @ x.reshape(-1)
        segs = [y[:self.nseg]]
        el = self.el
        for k in range(V.shape[0]):
            Th = y[self.nseg:]
            Rh = el.R0 * (1 + el.alpha * Th)
            P = V[k] ** 2 * Rh / (Rh + el.Rs) ** 2
            d = self.dec[k]
            x = d[None, :] * x + (1 - d)[None, :] * P[:, None]
            y = self.R2 @ x.reshape(-1)
            segs.append(y[:self.nseg])
        return torch.stack(segs)


def settling(t, e, eps):
    above = np.nonzero(e > eps)[0]
    if len(above) == 0:
        return 0.0
    k = above[-1]
    return float('inf') if k == len(e) - 1 else float(t[k + 1])


def design_mimo(dm, VA_P0, VB, Uss, opt, e_ref, iters=400, lr=0.05, Te=40e-6, verbose=False):
    """Gradient design.  u(t) = PWL knots every 1 us on [0, Te] (pinned to 0 at Te)
    + sum_k a_hk exp(-t/tau_k) over the identified poles; V = Vmax*sigmoid(logit(VB/Vmax)+u).
    Loss: log-time-weighted squared error relative to the static-compensation error e_ref."""
    P0 = VA_P0
    Vmax = dm.el.Vmax
    t = torch.tensor(TGRID[:-1])                        # ZOH sample times
    kt = torch.arange(0, Te + 1e-12, 1e-6)
    j = torch.clamp(torch.searchsorted(kt, t, right=True) - 1, 0, len(kt) - 2)
    wgt = ((t - kt[j]) / (kt[j + 1] - kt[j])).clamp(0, 1)[:, None]
    early = (t < kt[-1])[:, None]
    expo = torch.exp(-t[:, None] / dm.taus[None, :])    # (T-1, K)
    H = len(VB)
    nk = len(kt) - 1
    z = torch.zeros(nk + len(dm.taus), H, requires_grad=True)
    zB = torch.logit(torch.tensor(VB) / Vmax)

    def wave(z):
        uk = torch.cat([z[:nk], torch.zeros(1, H)], 0)
        u = torch.where(early, (1 - wgt) * uk[j] + wgt * uk[j + 1], torch.zeros(1, H)) + expo @ z[nk:]
        return Vmax * torch.sigmoid(zB[None, :] + u)

    tt = torch.tensor(TGRID)
    w = torch.tensor(np.gradient(np.log(TGRID + 2e-6)))   # log-time weights
    w = w / w.sum()
    eref = torch.tensor(e_ref)
    optm = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optm, iters)
    hist = []
    for it in range(iters):
        optm.zero_grad()
        seg = dm.run(wave(z), P0)
        e = opt.err(seg, Uss)
        L = (w * e ** 2 / (eref ** 2 + 1e-8)).sum()
        L.backward(); optm.step(); sched.step()
        hist.append(float(L.detach()))
        if verbose and it % 50 == 0:
            print(f'  it {it} loss {hist[-1]:.4e}', flush=True)
    with torch.no_grad():
        return wave(z).numpy(), hist


def design_siso(dm, opt, PA, PB, VB, TB_heat):
    """Per-heater bang-bang pre-emphasis (prior practice): each heater is overdriven (Vmax when
    its power rises, 0 when it falls) for a time t_on chosen so that its OWN element's phase,
    predicted from its own identified response only, best reaches the target; then the static
    voltage.  Crosstalk is handled only statically."""
    R = dm.R.numpy()[:dm.nseg]                   # (nseg, H, K)
    taus = dm.taus.numpy()
    H = len(VB)
    own = np.stack([opt.diff_phases(R[:, h, :].T)[:, h] for h in range(H)], 0)   # (H, K)
    t = TGRID
    w = np.gradient(np.log(t + 2e-6))
    Pod_up = dm.el.power(dm.el.Vmax, TB_heat)
    cand = np.concatenate([[0.0], TGRID[1:][TGRID[1:] <= 100e-6]])
    V = np.repeat(VB[None, :], len(t) - 1, 0)
    ton = np.zeros(H)
    Sown = (1 - np.exp(-t[:, None] / taus[None, :])) @ own.T                     # (T+1, H)
    for h in range(H):
        up = PB[h] > PA[h]
        Pod = Pod_up[h] if up else 0.0
        sh = t[None, :] - cand[:, None]                                          # (C, T+1)
        Ssh = ((1 - np.exp(-np.clip(sh, 0, None)[..., None] / taus)) @ own[h]) * (sh >= 0)
        ph = (Pod - PA[h]) * Sown[None, :, h] - (Pod - PB[h]) * Ssh
        tgt = (PB[h] - PA[h]) * own[h].sum()
        c = ((ph - tgt) ** 2 * w[None, :]).sum(1)
        best = cand[np.argmin(c)]
        ton[h] = best
        V[t[:-1] < best, h] = dm.el.Vmax if up else 0.0
    return V, ton


def random_targets(rng, M):
    return np.concatenate([rng.uniform(0, 2 * math.pi, M), rng.uniform(0, math.pi, M)])


def design_lbfgs(dm, PA, VB, Uss, opt, e_ref, basis_fn, outer=15, inner=20, delta=1e-4,
                 z0=None, verbose=False, Vbase=None):
    """Proposed design: minimise the exact (nonlinear) mesh error, time-weighted per decade and
    normalised by the static-compensation error, with L-BFGS (strong-Wolfe line search).
    Drive V = Vmax*sigmoid(logit(VB/Vmax) + G^T z): bounded by construction, equal to VB when
    z = 0, and every basis function in G decays, so V -> VB."""
    Vmax = dm.el.Vmax
    G = torch.tensor(basis_fn(TGRID[:-1]))                       # (B, T)
    H = len(VB)
    z = torch.zeros(G.shape[0], H) if z0 is None else torch.tensor(z0).clone()
    z.requires_grad_(True)
    if Vbase is None:
        zB = torch.logit(torch.tensor(VB) / Vmax)[None, :]
    else:   # start from a given waveform (e.g. per-heater pre-emphasis); it must end at VB
        zB = torch.logit(torch.tensor(np.clip(Vbase / Vmax, 0.01, 0.99)))
        tail = torch.tensor(TGRID[:-1] > 150e-6)[:, None]
        zB = torch.where(tail, torch.logit(torch.tensor(VB) / Vmax)[None, :], zB)
    lw = torch.tensor(np.gradient(np.log(TGRID + 1e-7)))
    w = lw / (torch.tensor(e_ref) ** 2 + delta ** 2)
    w = w / w.sum()

    def wave(z):
        return Vmax * torch.sigmoid(zB + G.T @ z)

    opt_ = torch.optim.LBFGS([z], lr=1.0, max_iter=inner, history_size=30,
                             line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-14)
    hist = []

    def closure():
        opt_.zero_grad()
        e = opt.err(dm.run(wave(z), PA), Uss)
        L = (w * e ** 2).sum()
        L.backward()
        hist.append(float(L.detach()))
        return L
    for o in range(outer):
        opt_.step(closure)
        if verbose:
            print(f'   L-BFGS outer {o}: loss {hist[-1]:.4e} ({len(hist)} evals)', flush=True)
    with torch.no_grad():
        return wave(z).numpy(), z.detach().numpy(), hist
