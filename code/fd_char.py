import time, json, numpy as np
from fd_thermal import FDMesh, time_grid
from fd_modal import ModalFD
def characterise(fd, label):
    md=ModalFD(fd); N=fd.lay.N
    h = fd.M + [i for i,(c,mm) in enumerate(fd.mzis) if c==3 and mm==3][0]
    hs=fd.heat_seg[h]; k,mm=divmod(hs,N)
    outs={'self':hs,'adj30':k*N+mm+1,'next60':k*N+mm+2,'far90':k*N+mm-3}
    P=np.zeros(fd.H); P[h]=1e-3
    t=time_grid(((100e-6,0.25e-6),(300e-6,1e-6),(1e-3,5e-6),(10e-3,20e-6)))
    S,Hh,_=md.simulate(t, lambda k_,T: P)
    fin,hfin=md.steady(P)
    r={'label':label,'modes':int(md.sig.size),'slowest_tau_ms':float(1/md.sig.min()*1e3)}
    b=fd.beta
    r['Ppi_mW']=float(np.pi/(b*(fin[hs]-fin[k*N+mm+1]))*1e-3)
    r['heater_K_per_mW']=float(hfin[h])
    for name,o in outs.items():
        y=S[:,o]/fin[o]
        def ta(f):
            i=np.nonzero(y>=f)[0]; return float(t[i[0]]*1e6) if len(i) else None
        r[name]={'ratio_pct':float(fin[o]/fin[hs]*100),'t63_us':ta(0.632),'t90_us':ta(0.9),'t99_us':ta(0.99),'t999_us':ta(0.999)}
    print(json.dumps(r,indent=1)); return r, t, S[:,list(outs.values())]/fin[list(outs.values())]
r1,t,Y=characterise(FDMesh(), 'base')
np.save('char_base.npy', np.c_[t,Y])
r2,_,_=characterise(FDMesh(dy_fine=1.25e-6, dz_top=0.25e-6), 'fine_yz')
r3,_,_=characterise(FDMesh(dx=12.5e-6), 'fine_x')
json.dump([r1,r2,r3], open('fd_char.json','w'), indent=1)
