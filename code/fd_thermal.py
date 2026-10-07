"""3-D finite-volume transient heat model of a thermo-optic Clements mesh on SOI.

Stack (z = 0 at the waveguide plane):
    z in [0, t_clad]            SiO2 top cladding; TiN heaters at z = z_h (adiabatic top surface)
    z in [-t_box, 0]            SiO2 buried oxide (the 220 nm Si device layer is thermally
                                negligible and treated as oxide)
    z in [-t_box - t_sub, -t_box]  Si substrate; isothermal bottom (heat sink, dT = 0)
Lateral faces adiabatic, placed far from the mesh.

Cell-centred finite volumes on a tensor grid (non-uniform in y and z), harmonic-mean face
conductances.  Time integration: variable-step BDF2 (backward Euler first step), linear solves by
algebraic-multigrid-preconditioned CG.  Inputs are zero-order-hold heater powers.

Outputs: waveguide-segment temperatures integrated along x over each optical interval of the
mesh (the quantity that sets the optical phase), and heater temperatures (for the TCR).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyamg


@dataclass
class Stack:
    k_si: float = 148.0          # W/m/K
    rc_si: float = 2330 * 705.0  # rho*c, J/m^3/K
    k_ox: float = 1.38
    rc_ox: float = 2203 * 745.0
    t_clad: float = 3.0e-6
    z_h: float = 2.0e-6          # heater height above waveguide plane
    t_box: float = 2.0e-6
    t_sub: float = 725e-6
    dndT: float = 1.86e-4        # silicon thermo-optic coefficient, 1/K
    wl: float = 1.55e-6


@dataclass
class Layout:
    N: int = 8
    pitch: float = 30e-6
    Lc: float = 300e-6
    Lh: float = 100e-6           # heater length
    ext_c: float = 0.15          # heater centres, fraction of Lc (ext and int slots)
    int_c: float = 0.55
    cpl1: float = 0.35           # coupler positions (interval boundaries), fraction of Lc
    cpl2: float = 0.75
    margin: float = 300e-6


def graded(a, b, h0, ratio, hmax):
    """Cells from a to b starting at size h0, growing by ratio up to hmax (returns edges)."""
    e = [a]
    h = h0
    while e[-1] + h < b - 1e-12:
        e.append(e[-1] + h)
        h = min(h * ratio, hmax)
    e.append(b)
    if len(e) > 2 and (e[-1] - e[-2]) < 0.5 * (e[-2] - e[-3]):
        e.pop(-2)
    return np.array(e)


class FDMesh:
    def __init__(self, st=Stack(), lay=Layout(), dx=25e-6, dy_fine=2.5e-6, dz_top=0.5e-6,
                 z_ratio=1.6, y_ratio=1.5, scale=None):
        self.st, self.lay = st, lay
        N, p = lay.N, lay.pitch
        # ---- x: uniform
        x0, x1 = -lay.margin, N * lay.Lc + lay.margin
        nx = int(round((x1 - x0) / dx))
        self.xe = np.linspace(x0, x1, nx + 1)
        # ---- y: fine band (+-7.5 um) around each waveguide, 5 um between, graded margins
        ye = []
        for m in range(N):
            yc = m * p
            band = np.arange(yc - 7.5e-6, yc + 7.5e-6 + 1e-12, dy_fine)
            ye += list(band)
            if m < N - 1:
                gap = np.linspace(yc + 7.5e-6, yc + p - 7.5e-6, int(round((p - 15e-6) / 5e-6)) + 1)
                ye += list(gap[1:-1])
        ye = np.unique(np.round(np.array(ye), 12))
        lo = graded(ye[0], ye[0] + lay.margin, dy_fine * 2, y_ratio, 60e-6)
        hi = graded(ye[-1], ye[-1] + lay.margin, dy_fine * 2, y_ratio, 60e-6)
        self.ye = np.unique(np.concatenate([ye[0] - (lo - ye[0])[::-1], ye, hi]))
        # ---- z: cladding and BOX uniform dz_top, substrate graded downwards
        zc = np.arange(0, st.t_clad + 1e-12, dz_top)
        zb = np.arange(-st.t_box, 0, dz_top)
        zs = graded(0, st.t_sub, dz_top, z_ratio, 1e9)
        zsub = -st.t_box - zs[::-1]
        self.ze = np.unique(np.round(np.concatenate([zsub, zb, zc]), 12))
        self.build(scale)

    # ------------------------------------------------------------------
    def build(self, scale=None):
        st, lay = self.st, self.lay
        s = scale or {}
        k_ox = st.k_ox * s.get('k_ox', 1.0)
        k_si = st.k_si * s.get('k_si', 1.0)
        rc_ox = st.rc_ox * s.get('rc_ox', 1.0)
        xe, ye, ze = self.xe, self.ye, self.ze
        self.xc, self.yc, self.zc = [(e[1:] + e[:-1]) / 2 for e in (xe, ye, ze)]
        dx, dy, dz = [np.diff(e) for e in (xe, ye, ze)]
        nx, ny, nz = len(dx), len(dy), len(dz)
        self.shape = (nx, ny, nz)
        n = nx * ny * nz
        self.n = n
        kz = np.where(self.zc < -st.t_box, k_si, k_ox)
        rcz = np.where(self.zc < -st.t_box, st.rc_si, rc_ox)
        idx = np.arange(n).reshape(nx, ny, nz)
        K = kz[None, None, :] * np.ones((nx, ny, nz))
        vol = dx[:, None, None] * dy[None, :, None] * dz[None, None, :]
        self.Cdiag = (rcz[None, None, :] * vol).ravel()
        rows, cols, vals = [], [], []

        def add(i, j, g):
            rows.extend([i, j]); cols.extend([j, i]); vals.extend([-g, -g])
        diag = np.zeros(n)
        # x faces
        A = dy[None, :, None] * dz[None, None, :]
        ka, kb = K[:-1], K[1:]
        g = A / (dx[:-1, None, None] / (2 * ka) + dx[1:, None, None] / (2 * kb))
        i, j = idx[:-1].ravel(), idx[1:].ravel(); g = g.ravel()
        rows += [i, j]; cols += [j, i]; vals += [-g, -g]
        np.add.at(diag, i, g); np.add.at(diag, j, g)
        # y faces
        A = dx[:, None, None] * dz[None, None, :]
        ka, kb = K[:, :-1], K[:, 1:]
        g = A / (dy[None, :-1, None] / (2 * ka) + dy[None, 1:, None] / (2 * kb))
        i, j = idx[:, :-1].ravel(), idx[:, 1:].ravel(); g = g.ravel()
        rows += [i, j]; cols += [j, i]; vals += [-g, -g]
        np.add.at(diag, i, g); np.add.at(diag, j, g)
        # z faces
        A = dx[:, None, None] * dy[None, :, None]
        ka, kb = K[:, :, :-1], K[:, :, 1:]
        g = A / (dz[None, None, :-1] / (2 * ka) + dz[None, None, 1:] / (2 * kb))
        i, j = idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel(); g = g.ravel()
        rows += [i, j]; cols += [j, i]; vals += [-g, -g]
        np.add.at(diag, i, g); np.add.at(diag, j, g)
        # isothermal bottom
        gb = (dx[:, None] * dy[None, :]) * K[:, :, 0] / (dz[0] / 2)
        np.add.at(diag, idx[:, :, 0].ravel(), gb.ravel())
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
        self.G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n, n))
        self._build_io()

    # ------------------------------------------------------------------
    def _build_io(self):
        st, lay = self.st, self.lay
        N = lay.N
        nx, ny, nz = self.shape
        idx = np.arange(self.n).reshape(nx, ny, nz)
        dx = np.diff(self.xe)
        kz_h = np.argmin(abs(self.zc - st.z_h))
        kz_w = np.argmin(abs(self.zc - 0.0 + 1e-9))  # cell just above/below z=0: take nearest
        kz_w = np.argmin(abs(self.zc))
        self.mzis = [(c, m) for c in range(N) for m in range(c % 2, N - 1, 2)]
        self.M = len(self.mzis)
        # heaters: ext slot then int slot, upper arm (mode m)
        heaters = [(c, m, lay.ext_c) for (c, m) in self.mzis] + [(c, m, lay.int_c) for (c, m) in self.mzis]
        self.H = len(heaters)
        Qr, Qc, Qv = [], [], []
        Hr, Hc, Hv = [], [], []
        for h, (c, m, frac) in enumerate(heaters):
            xc0 = c * lay.Lc + frac * lay.Lc
            xa, xb = xc0 - lay.Lh / 2, xc0 + lay.Lh / 2
            ov = np.clip(np.minimum(self.xe[1:], xb) - np.maximum(self.xe[:-1], xa), 0, None)
            w = ov / ov.sum()
            jy = np.argmin(abs(self.yc - m * lay.pitch))
            for ix in np.nonzero(w)[0]:
                Qr.append(idx[ix, jy, kz_h]); Qc.append(h); Qv.append(w[ix])
                Hr.append(h); Hc.append(idx[ix, jy, kz_h]); Hv.append(w[ix])
        self.Q = sp.csr_matrix((Qv, (Qr, Qc)), shape=(self.n, self.H))       # power -> cell heat
        self.Hout = sp.csr_matrix((Hv, (Hr, Hc)), shape=(self.H, self.n))    # mean heater temp
        # optical intervals: for each column c, slot s (0 ext, 1 int), mode m: x-integral of dT
        cp = []
        for c in range(N):
            x0 = c * lay.Lc
            cp.append((x0 + (lay.cpl2 - 1) * lay.Lc, x0 + lay.cpl1 * lay.Lc))   # ext interval
            cp.append((x0 + lay.cpl1 * lay.Lc, x0 + lay.cpl2 * lay.Lc))         # int interval
        cp[0] = (-lay.margin * 0.5, cp[0][1])            # input leads
        cp[-1] = (cp[-1][0], cp[-1][1])
        Or, Oc, Ov = [], [], []
        for k, (xa, xb) in enumerate(cp):
            ov = np.clip(np.minimum(self.xe[1:], xb) - np.maximum(self.xe[:-1], xa), 0, None)
            for m in range(N):
                jy = np.argmin(abs(self.yc - m * lay.pitch))
                o = k * N + m
                for ix in np.nonzero(ov)[0]:
                    Or.append(o); Oc.append(idx[ix, jy, kz_w]); Ov.append(ov[ix])
        self.nseg = 2 * N * N
        self.Seg = sp.csr_matrix((Ov, (Or, Oc)), shape=(self.nseg, self.n))  # integral dT dx [K m]
        self.beta = 2 * math.pi / st.wl * st.dndT                              # rad / (K m)
        self.heat_seg = np.array([ (2 * c + (0 if frac == lay.ext_c else 1)) * N + m
                                   for (c, m, frac) in heaters])

    # ------------------------------------------------------------------
    def steady(self, P, tol=1e-10):
        ml = self._ml(None)
        x, info = spla.cg(self.G, self.Q @ P, rtol=tol, maxiter=500, M=ml.aspreconditioner())
        assert info == 0
        return x

    def _ml(self, a):
        key = a
        if not hasattr(self, '_mls'):
            self._mls = {}
        if key not in self._mls:
            Amat = self.G if a is None else (self.G + sp.diags(a * self.Cdiag)).tocsr()
            self._mls[key] = (Amat, pyamg.smoothed_aggregation_solver(Amat, symmetry='symmetric'))
        return self._mls[key][1]

    def transient(self, tgrid, Pfun, x0=None, tol=1e-9, record_every=1):
        """Variable-step BDF2.  tgrid: increasing times (t[0]=0).  Pfun(k, x) -> heater powers
        applied on (t[k], t[k+1]] (ZOH, may depend on the state for the TCR).
        Returns segment integrals (len(t), nseg) and heater temps (len(t), H)."""
        x = np.zeros(self.n) if x0 is None else x0.copy()
        segs = [self.Seg @ x]; heat = [self.Hout @ x]
        xprev, hprev = None, None
        for k in range(len(tgrid) - 1):
            h = tgrid[k + 1] - tgrid[k]
            P = Pfun(k, heat[-1])
            if xprev is None:                     # backward Euler
                a = 1.0 / h
                rhs = self.Cdiag * x * a + self.Q @ P
            else:                                 # variable-step BDF2
                w = h / hprev
                a0 = (1 + 2 * w) / (1 + w)
                a1 = -(1 + w)
                a2 = w * w / (1 + w)
                a = a0 / h
                rhs = self.Cdiag * (-(a1 * x + a2 * xprev)) / h + self.Q @ P
            key = round(a, 3)
            ml = self._ml(key)
            Amat = self._mls[key][0]
            xn, info = spla.cg(Amat, rhs, x0=x, rtol=tol, maxiter=300, M=ml.aspreconditioner())
            assert info == 0, info
            xprev, hprev, x = x, h, xn
            segs.append(self.Seg @ x); heat.append(self.Hout @ x)
        return np.array(segs), np.array(heat), x


def time_grid(segments=((100e-6, 0.25e-6), (300e-6, 1e-6), (1e-3, 5e-6), (3e-3, 20e-6))):
    """Piecewise-uniform time grid: few distinct steps, so few multigrid set-ups."""
    t = [0.0]
    for t_end, h in segments:
        n = int(round((t_end - t[-1]) / h))
        t += list(t[-1] + h * np.arange(1, n + 1))
    return np.array(t)
