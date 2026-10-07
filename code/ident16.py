"""Step responses and compact model (K = 24) for the 16x16 mesh (240 heaters)."""
import time, json
import numpy as np
from fd_thermal import FDMesh, Layout
from fd_modal import ModalFD
from ident import TGRID
from ident_fast import step_responses_fast
from drive import TAUS
import os
os.makedirs('out', exist_ok=True)

t0 = time.time()
fd = FDMesh(lay=Layout(N=16)); md = ModalFD(fd)
Y = step_responses_fast(md, TGRID, dtype=np.float32)
print('steps %.0f s' % (time.time() - t0), Y.shape, flush=True)
Sg, Hg = md.dc_gains()
dc = np.concatenate([Sg, Hg], 0)
# chunked version of ident.fit (identical algebra): R_k = pinv(A w) (b w), DC enforced
t = TGRID; K = len(TAUS)
Phi = 1 - np.exp(-t[:, None] / TAUS[None, :])
A = Phi[:, :-1] - Phi[:, -1:]
w = np.sqrt(np.gradient(np.log(t + 1e-7)))[:, None]
Pinv = np.linalg.pinv(A * w, rcond=1e-12)
T, O, H = Y.shape
R = np.zeros((O, H, K))
maxrel = 0.0
for o in range(O):
    y = Y[:, o, :].astype(np.float64); d = dc[o]
    Rk = Pinv @ ((y - Phi[:, -1:] * d[None, :]) * w)
    R[o, :, :-1] = Rk.T; R[o, :, -1] = d - Rk.sum(0)
    pred = Phi @ R[o].T
    if o < fd.nseg:
        maxrel = max(maxrel, np.abs(pred - y).max())
np.save('R16.npy', R)
selfmax = np.abs(dc[:fd.nseg]).max()
info = {'cells': list(fd.shape), 'n': int(fd.n), 'H': int(fd.H), 'steps_s': time.time() - t0,
        'max_step_err_rel_to_self_pct': float(maxrel / selfmax * 100)}
json.dump(info, open('out/ident16.json', 'w'), indent=1)
print(info, flush=True)
