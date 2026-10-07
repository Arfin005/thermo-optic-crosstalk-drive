"""Reference step responses of every heater and identification of a compact design model.

Design model (shared real poles, per heater):
    y_o(t) = sum_h sum_k R[o, h, k] x_{h,k}(t),   tau_k dx_{h,k}/dt = -x_{h,k} + P_h(t)
Poles tau_k are fixed on a logarithmic grid; residues come from linear least squares on the
sampled step responses, with the DC gain enforced exactly.  Optionally the step responses are
corrupted by measurement noise first (per-chip identification scenario).
"""
import sys, json, time
import numpy as np
from fd_thermal import FDMesh, time_grid
from fd_modal import ModalFD

TGRID = time_grid(((100e-6, 0.25e-6), (300e-6, 1e-6), (1e-3, 5e-6), (10e-3, 20e-6)))


def step_responses(md, t=TGRID, P0=1e-3):
    """Segment integrals and heater temps per watt for a step on each heater: (T, O+H, H)."""
    H = md.H
    Y = np.zeros((len(t), md.O + H, H))
    for h in range(H):
        P = np.zeros(H); P[h] = P0
        S, Hh, _ = md.simulate(t, lambda k, T: P)
        Y[:, :md.O, h] = S / P0
        Y[:, md.O:, h] = Hh / P0
    return Y


def fit(Y, t, taus, dc=None, rcond=1e-12):
    """Least-squares residues for fixed poles; the DC gain sum_k R = dc is enforced exactly."""
    T, O, H = Y.shape
    K = len(taus)
    Phi = 1 - np.exp(-t[:, None] / taus[None, :])          # (T, K)
    Yf = Y.reshape(T, O * H)
    dcv = (Y[-1] if dc is None else dc).reshape(O * H)
    # eliminate last residue: R_K = dc - sum_{k<K} R_k  -> (Phi_k - Phi_K) R_k = Y - dc*Phi_K
    A = Phi[:, :-1] - Phi[:, -1:]
    b = Yf - Phi[:, -1:] * dcv[None, :]
    # weight samples uniformly in log-time so every decade counts
    w = np.sqrt(np.gradient(np.log(t + 1e-7)))[:, None]
    Rk, *_ = np.linalg.lstsq(A * w, b * w, rcond=rcond)
    R = np.vstack([Rk, dcv[None, :] - Rk.sum(0, keepdims=True)])  # (K, O*H)
    return R.T.reshape(O, H, K)


def predict(R, t, taus):
    Phi = 1 - np.exp(-t[:, None] / taus[None, :])
    return np.einsum('tk,ohk->toh', Phi, R)


if __name__ == '__main__':
    t0 = time.time()
    fd = FDMesh(); md = ModalFD(fd)
    Y = step_responses(md)
    Sg, Hg = md.dc_gains()
    dc = np.concatenate([Sg, Hg], 0)
    print('step responses %.0fs' % (time.time() - t0), Y.shape, flush=True)
    np.save('ref_steps.npy', Y); np.save('ref_dc.npy', dc)
    beta = fd.beta
    out = {}
    for K in (6, 8, 10, 12, 16):
        taus = np.logspace(np.log10(1e-6), np.log10(5e-3), K)
        R = fit(Y, TGRID, taus, dc=dc)
        E = predict(R, TGRID, taus) - Y
        # phase error for a 25 mW step, segments only [rad]
        ph_err = beta * np.abs(E[:, :fd.nseg]).max() * 25e-3
        self_ph = beta * np.abs(dc[:fd.nseg]).max() * 25e-3
        cross = beta * np.abs(E[:, :fd.nseg]).max(axis=(0,))  # per (o,h)
        out[K] = {'max_phase_err_mrad_25mW': float(ph_err * 1e3),
                  'max_err_rel_to_self_pct': float(ph_err / self_ph * 100)}
        print(K, out[K], flush=True)
    json.dump(out, open('ident_fit.json', 'w'), indent=1)
