import os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
os.makedirs('tex/figs', exist_ok=True)
C=['#2a78d6','#eb6834','#1baf7a','#eda100']
plt.rcParams.update({'font.size':8,'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'mathtext.fontset':'stix','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
  'axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.5,'lines.linewidth':1.3,'savefig.dpi':300})
# ---- (a) layout top view, (b) cross-section
fig=plt.figure(figsize=(7.16,2.3))
ax=fig.add_axes([0.08,0.18,0.59,0.71])
N=8; Lc=300; p=30
for m in range(N): ax.plot([-60,N*Lc+20],[m*p,m*p],color='#8a8a85',lw=0.6)
for c in range(N):
    x0=c*Lc
    for m in range(c%2,N-1,2):
        for f in (0.35,0.75):
            ax.plot([x0+f*Lc-12,x0+f*Lc+12],[m*p+3,(m+1)*p-3],color='#1a1a19',lw=0.7)
            ax.plot([x0+f*Lc-12,x0+f*Lc+12],[(m+1)*p-3,m*p+3],color='#1a1a19',lw=0.7)
        for f,col in ((0.15,C[0]),(0.55,C[1])):
            ax.add_patch(Rectangle((x0+f*Lc-50,m*p-4),100,8,color=col,lw=0))
ax.set_xlim(-80,N*Lc+30); ax.set_ylim(-20,(N-1)*p+20); ax.set_aspect('auto')
ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)'); ax.grid(False)
ax.set_title('(a) 8×8 Clements mesh: external (blue) and internal (orange) heaters',loc='left',fontsize=8)
ax2=fig.add_axes([0.745,0.18,0.235,0.71])
ax2.add_patch(Rectangle((-20,-12),40,10,color='#c3c2b7')); ax2.text(0,-9,'Si substrate 725 µm\n(isothermal bottom)',ha='center',va='center',fontsize=6.5)
ax2.add_patch(Rectangle((-20,-2),40,2,color='#dfe9f7')); ax2.text(13,-1,'BOX 2 µm',ha='center',va='center',fontsize=6.5)
ax2.add_patch(Rectangle((-20,0),40,3,color='#edf3fb')); ax2.text(13,1.5,'cladding\n3 µm',ha='center',va='center',fontsize=6.5)
ax2.add_patch(Rectangle((-0.25,-0.0),0.5,0.22,color='#1a1a19')); ax2.text(-2.5,0.1,'waveguide',fontsize=6.5,ha='right',va='center')
ax2.add_patch(Rectangle((-1,1.7),2,0.12,color=C[1])); ax2.text(-2.5,1.76,'TiN heater',fontsize=6.5,ha='right',va='center')
ax2.set_xlim(-20,20); ax2.set_ylim(-12,3.5); ax2.set_xlabel('y (µm)'); ax2.set_ylabel('z (µm)'); ax2.grid(False)
ax2.set_title('(b) Cross-section (not to scale)',loc='left',fontsize=8)
fig.savefig('tex/figs/fig_layout.pdf'); plt.close(fig)
# ---- step responses on log time
d=np.load('char_base.npy'); t=d[:,0]; Y=d[:,1:]
fig,ax=plt.subplots(figsize=(3.5,2.3))
labs=['Heated waveguide','30 µm away','60 µm away','90 µm away']
for i,(l,ls) in enumerate(zip(labs,['-','--','-.',':'])):
    ax.semilogx(t[1:]*1e6,Y[1:,i],color=C[i],ls=ls,label=l)
ax.set_xlim(0.25,1e4); ax.set_ylim(0,1.02)
ax.set_xlabel('Time after 1 mW heater step (µs)'); ax.set_ylabel('Normalised phase change')
ax.legend(frameon=False,fontsize=6.5,loc='lower right')
fig.tight_layout(); fig.savefig('tex/figs/fig_steps.pdf'); plt.close(fig)
print('ok')
