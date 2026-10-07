import os, json, numpy as np, torch
torch.set_num_threads(1)
from drive import *
from ident import fit, predict
chip=Chip(); Y=np.load('ref_steps.npy'); dc=np.load('ref_dc.npy'); nseg=chip.nseg; b=chip.fd.beta
rows=[]
cases=[]
for s in (1,2,3):
    rng=np.random.default_rng(s); M=chip.fd.M
    tA,tB=random_targets(rng,M),random_targets(rng,M); PA,PB=chip.static_powers(tA),chip.static_powers(tB)
    VB=chip.static_volts(PB); T=len(TGRID)-1; V=np.repeat(VB[None],T,0)
    cases.append((PA,V,chip.run(V,PA)))
for K in (8,12,16,24):
    taus=np.logspace(np.log10(0.25e-6),np.log10(2.4e-3),K)
    R=fit(Y,TGRID,taus,dc=dc); E=predict(R,TGRID,taus)-Y
    step=np.abs(E[:,:nseg]).max()/np.abs(dc[:nseg]).max()*100
    dm=DesignModel(R,taus,nseg,chip.el); ph=0
    for PA,V,S in cases:
        with torch.no_grad(): Sm=dm.run(torch.tensor(V),PA).numpy()
        ph=max(ph,(b*np.abs(S-Sm)).max()*1e3)
    rows.append({'K':K,'states':K*chip.H,'step_err_pct':float(step),'phase_err_mrad':float(ph)}); print(rows[-1],flush=True)
os.makedirs('out', exist_ok=True)
json.dump(rows,open('out/tab_rom.json','w'),indent=1)
os.makedirs('tex', exist_ok=True)
with open('tex/tab_rom.tex','w') as f:
    f.write('\\begin{table}[t]\n\\centering\n\\caption{Accuracy of the Compact Model Against the Three-Dimensional Model}\n\\label{tab:rom}\n')
    f.write('\\begin{tabular}{@{}rrrr@{}}\n\\toprule\n$K$ & States & Max.\\ step error & Max.\\ phase error\\\\\n & & (\\% of self) & in reconfiguration (mrad)\\\\\n\\midrule\n')
    for r in rows: f.write(f"{r['K']} & {r['states']} & {r['step_err_pct']:.3f} & {r['phase_err_mrad']:.2f}\\\\\n")
    f.write('\\bottomrule\n\\end{tabular}\n\\\\[2pt]{\\footnotesize Phase error: largest interval-phase difference over three static-step reconfigurations, \\SI{10}{ms}.}\n\\end{table}\n')
