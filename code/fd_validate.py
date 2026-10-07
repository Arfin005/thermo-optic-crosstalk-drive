import time, numpy as np
from fd_thermal import FDMesh, time_grid
from fd_modal import ModalFD
fd=FDMesh(); N=fd.lay.N
t0=time.time(); md=ModalFD(fd); print('modal build %.1fs, modes %d'%(time.time()-t0, md.sig.size))
h = fd.M + [i for i,(c,mm) in enumerate(fd.mzis) if c==3 and mm==3][0]
hs=fd.heat_seg[h]; k,mm=divmod(hs,N); outs=[hs,k*N+mm+1,k*N+mm+2]
P=np.zeros(fd.H); P[h]=1e-3
seg_m,heat_m=md.steady(P)
x=fd.steady(P, tol=1e-12); seg_c=fd.Seg@x; heat_c=fd.Hout@x
print('steady: modal', seg_m[outs], heat_m[h]); print('steady: CG   ', seg_c[outs], heat_c[h])
print('max rel diff seg %.2e'%(np.abs(seg_m-seg_c).max()/np.abs(seg_c).max()))
# transient: 40 us at 0.25 us, compare with CG BDF2
t=np.arange(0,40e-6+1e-15,0.25e-6)
t0=time.time(); Sm,Hm,_=md.simulate(t, lambda k_,T: P); print('modal transient %.2fs'%(time.time()-t0))
t0=time.time(); Sc,Hc,_=fd.transient(t, lambda k_,T: P, tol=1e-10); print('CG BDF2 transient %.1fs'%(time.time()-t0))
for i,o in enumerate(outs):
    print(' out',o,'max |diff|/final %.2e'%(np.abs(Sm[:,o]-Sc[:,o]).max()/abs(seg_m[o])))
