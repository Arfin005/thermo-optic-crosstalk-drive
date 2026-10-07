import numpy as np, torch
from drive import *
from ident import fit, predict
chip=Chip(); Y=np.load('ref_steps.npy'); dc=np.load('ref_dc.npy')
rng=np.random.default_rng(1); M=chip.fd.M
tA,tB=random_targets(rng,M),random_targets(rng,M)
PA,PB=chip.static_powers(tA),chip.static_powers(tB); VB=chip.static_volts(PB)
Uss=chip.opt.unitary(torch.tensor(chip.Sg@PB)); T=len(TGRID)-1
V=np.repeat(VB[None],T,0); S=chip.run(V,PA); er=chip.opt.err(torch.tensor(S),Uss).numpy()
print('slowest ref taus (ms):', np.sort(1/chip.md.sig.ravel())[::-1][:5]*1e3)
for (lo,hi,K) in [(0.25e-6,2.4e-3,24),(0.25e-6,2.4e-3,32),(0.1e-6,2.4e-3,40)]:
    taus=np.logspace(np.log10(lo),np.log10(hi),K)
    R=fit(Y,TGRID,taus,dc=dc); dm=DesignModel(R,taus,chip.nseg,chip.el)
    with torch.no_grad(): Sm=dm.run(torch.tensor(V),PA).numpy()
    em=chip.opt.err(torch.tensor(Sm),Uss).numpy()
    E=predict(R,TGRID,taus)-Y
    vals=tuple(x for tt in [300e-6,1e-3,3e-3,1e-2] for x in (er[min(np.searchsorted(TGRID,tt),T)],em[min(np.searchsorted(TGRID,tt),T)]))
    print(lo,hi,K,'stepfit max rel %.3f%%'%(np.abs(E[:,:chip.nseg]).max()/np.abs(dc[:chip.nseg]).max()*100),
      ' ref/model 300us %.2e/%.2e 1ms %.2e/%.2e 3ms %.2e/%.2e 10ms %.2e/%.2e'%vals,
      ' max dphase %.4f rad'%(chip.fd.beta*np.abs(S-Sm)).max(), flush=True)
