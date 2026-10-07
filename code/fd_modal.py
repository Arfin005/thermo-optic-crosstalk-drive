"""Exact modal solution of the 3-D finite-volume heat model in fd_thermal.FDMesh.

Because the materials depend only on z and the grid is a tensor product with a uniform, adiabatic
x-direction, the semi-discrete system  C dT/dt = -G T + Q P  separates exactly:
    x: discrete-cosine modes (uniform Neumann Laplacian),
    y: generalised eigenvectors of the weighted Neumann Laplacian,
    z: for each (x, y) mode pair, a 1-D generalised eigenproblem (tridiagonal, n_z unknowns).
Every one of the nx*ny*nz modes then obeys  dc/dt = -sigma c + b^T P,  which is integrated
exactly for zero-order-hold inputs over any step.  No time-stepping error; no linear solves.
"""
import numpy as np
import scipy.linalg as sla


class ModalFD:
    def __init__(self, fd, scale=None):
        st = fd.st
        s = scale or {}
        k_ox = st.k_ox * s.get('k_ox', 1.0)
        k_si = st.k_si * s.get('k_si', 1.0)
        rc_ox = st.rc_ox * s.get('rc_ox', 1.0)
        self.fd = fd
        xe, ye, ze = fd.xe, fd.ye, fd.ze
        dx = np.diff(xe); dy = np.diff(ye); dz = np.diff(ze)
        assert np.allclose(dx, dx[0])
        dx = dx[0]
        nx, ny, nz = len(xe) - 1, len(dy), len(dz)
        zc = (ze[1:] + ze[:-1]) / 2
        kk = np.where(zc < -st.t_box, k_si, k_ox)
        rc = np.where(zc < -st.t_box, st.rc_si, rc_ox)
        # x modes
        Lx = 2 * np.eye(nx) - np.eye(nx, k=1) - np.eye(nx, k=-1)
        Lx[0, 0] = Lx[-1, -1] = 1
        mu, Phx = np.linalg.eigh(Lx / dx ** 2)
        # y modes: Ly phi = nu Dy phi
        w = 2.0 / (dy[:-1] + dy[1:])
        Ly = np.zeros((ny, ny))
        for j in range(ny - 1):
            Ly[j, j] += w[j]; Ly[j + 1, j + 1] += w[j]; Ly[j, j + 1] -= w[j]; Ly[j + 1, j] -= w[j]
        nu, Phy = sla.eigh(Ly, np.diag(dy))
        # z operator
        gam = 1.0 / (dz[:-1] / (2 * kk[:-1]) + dz[1:] / (2 * kk[1:]))
        Lz = np.zeros((nz, nz))
        for k in range(nz - 1):
            Lz[k, k] += gam[k]; Lz[k + 1, k + 1] += gam[k]; Lz[k, k + 1] -= gam[k]; Lz[k + 1, k] -= gam[k]
        Lz[0, 0] += kk[0] / (dz[0] / 2)          # isothermal bottom
        Kz = kk * dz
        Cz = rc * dz
        lam = (mu[:, None] + nu[None, :])       # (nx, ny)
        ci = 1 / np.sqrt(Cz)
        Asym = ci[:, None] * Lz * ci[None, :]
        M = Asym[None, None] + lam[..., None, None] * np.diag(Kz * ci * ci)[None, None]
        sig, V = np.linalg.eigh(M)               # (nx, ny, nz), (nx, ny, nz, nz)
        Psi = ci[None, None, :, None] * V        # Cz-orthonormal eigenvectors, [p,q,k,r]
        self.sig = sig
        self.dx = dx
        # heater and output geometry (from fd)
        idx = np.arange(fd.n).reshape(nx, ny, nz)
        ii, jj, kk_ = np.unravel_index(np.arange(fd.n), (nx, ny, nz))
        # heater input: each heater occupies cells (i, jh, kh) with weights w_i
        Q = fd.Q.tocsc()
        H = Q.shape[1]
        self.H = H
        Xh = np.zeros((nx, H)); jh = np.zeros(H, int); kh = np.zeros(H, int)
        for h in range(H):
            rows = Q.indices[Q.indptr[h]:Q.indptr[h + 1]]
            vals = Q.data[Q.indptr[h]:Q.indptr[h + 1]]
            jh[h] = jj[rows[0]]; kh[h] = kk_[rows[0]]
            assert np.all(jj[rows] == jh[h]) and np.all(kk_[rows] == kh[h])
            Xh[ii[rows], h] = vals
        self.Fx = Phx.T @ Xh / dx                # (nx_modes, H): x projection of heater source
        self.Fy = Phy[jh, :].T                   # (ny_modes, H)
        assert np.all(kh == kh[0]); self.kh = kh[0]
        self.Psi_h = Psi[:, :, self.kh, :]       # forcing into z-modes
        # outputs: segment integrals at (i, j_o, k_w) with weights ov_i ; heater mean temps
        S = fd.Seg.tocsr()
        O = S.shape[0]
        Xo = np.zeros((nx, O)); jo = np.zeros(O, int); ko = np.zeros(O, int)
        for o in range(O):
            cols = S.indices[S.indptr[o]:S.indptr[o + 1]]
            vals = S.data[S.indptr[o]:S.indptr[o + 1]]
            jo[o] = jj[cols[0]]; ko[o] = kk_[cols[0]]
            assert np.all(jj[cols] == jo[o])
            Xo[ii[cols], o] = vals
        assert np.all(ko == ko[0]); self.kw = ko[0]
        self.Gx_seg = Phx.T @ Xo                 # (nx, O)
        self.Gy_seg = Phy[jo, :].T               # (ny, O)
        self.Psi_w = Psi[:, :, self.kw, :]
        HO = fd.Hout.tocsr()
        Xho = np.zeros((nx, H))
        for h in range(H):
            cols = HO.indices[HO.indptr[h]:HO.indptr[h + 1]]
            Xho[ii[cols], h] = HO.data[HO.indptr[h]:HO.indptr[h + 1]]
        self.Gx_h = Phx.T @ Xho
        self.Gy_h = self.Fy
        self.O = O

    # ---- forcing of every mode by heater powers P (H,) -> (nx, ny, nz)
    def forcing(self, P):
        F = (self.Fx * P[None, :]) @ self.Fy.T    # (nx, ny)
        return self.Psi_h * F[..., None]

    def outputs(self, c):
        Zs = np.einsum('pqr,pqr->pq', self.Psi_w, c)
        Zh = np.einsum('pqr,pqr->pq', self.Psi_h, c)
        seg = (self.Gx_seg * (Zs @ self.Gy_seg)).sum(0)
        heat = (self.Gx_h * (Zh @ self.Gy_h)).sum(0)
        return seg, heat

    def steady(self, P):
        return self.outputs(self.forcing(P) / self.sig)

    def dc_gains(self):
        """Steady segment integrals and heater temps per watt: (O,H), (H,H)."""
        Sg = np.zeros((self.O, self.H)); Hg = np.zeros((self.H, self.H))
        for h in range(self.H):
            e = np.zeros(self.H); e[h] = 1.0
            Sg[:, h], Hg[:, h] = self.steady(e)
        return Sg, Hg

    def simulate(self, tgrid, Pfun, c0=None):
        """Exact ZOH integration.  Pfun(k, heater_temps) -> P on [t_k, t_k+1)."""
        c = np.zeros_like(self.sig) if c0 is None else c0.copy()
        seg, heat = self.outputs(c)
        S, Hh = [seg], [heat]
        cache = {}
        for k in range(len(tgrid) - 1):
            h = tgrid[k + 1] - tgrid[k]
            key = round(h * 1e12)
            if key not in cache:
                e = np.exp(-self.sig * h)
                cache[key] = (e, (1 - e) / self.sig)
            e, g = cache[key]
            P = Pfun(k, Hh[-1])
            c = e * c + g * self.forcing(P)
            seg, heat = self.outputs(c)
            S.append(seg); Hh.append(heat)
        return np.array(S), np.array(Hh), c

    def steady_state_modes(self, P):
        return self.forcing(P) / self.sig
