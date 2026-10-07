"""Fast exact step responses of every heater for the modal model (used for large meshes).

For a unit step on heater h from rest, every mode obeys c(t) = (1 - exp(-sigma t))/sigma * f_h, and
the forcing f_h and the outputs are separable in x, y and z.  Summing over the z-modes first and
contracting x and y with matrix products gives the same responses as ModalFD.simulate, far faster.
"""
import numpy as np


def step_responses_fast(md, t, dtype=np.float32, tchunk=64, out=None):
    """Segment integrals and heater temperatures per watt: array (T, O+H, H)."""
    O, H = md.O, md.H
    T = len(t)
    Y = np.zeros((T, O + H, H), dtype) if out is None else out
    for t0 in range(0, T, tchunk):
        tt = t[t0:t0 + tchunk]
        g = -np.expm1(-md.sig[None] * tt[:, None, None, None]) / md.sig[None]      # (tc, nx, ny, nz)
        Aw = np.einsum('pqr,tpqr->tpq', md.Psi_w * md.Psi_h, g)
        Ah = np.einsum('pqr,tpqr->tpq', md.Psi_h * md.Psi_h, g)
        del g
        for h in range(H):
            Bw = Aw * (md.Fx[:, h][None, :, None] * md.Fy[:, h][None, None, :])
            Bh = Ah * (md.Fx[:, h][None, :, None] * md.Fy[:, h][None, None, :])
            Y[t0:t0 + len(tt), :O, h] = np.einsum('tpo,po->to', Bw @ md.Gy_seg, md.Gx_seg)
            Y[t0:t0 + len(tt), O:, h] = np.einsum('tph,ph->th', Bh @ md.Gy_h, md.Gx_h)
    return Y
