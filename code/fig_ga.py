"""Graphical abstract (1320 x 590 px)."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
os.makedirs('tex', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'font.size': 9})
fig = plt.figure(figsize=(6.6, 2.95), dpi=300)
ax0 = fig.add_axes([0.0, 0.0, 0.40, 1.0]); ax0.axis('off'); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
boxes = [(0.865, '3-D thermal model of the mesh\n(exact modal solution)'),
         (0.622, 'Compact 24-pole model\nfrom heater step responses'),
         (0.378, 'Joint, history-aware drive design\n(exact matrix error, L-BFGS)'),
         (0.135, 'Faster reprogramming,\nno change to the chip')]
cols = ['#e8f0fb', '#e8f0fb', '#e3f5ee', '#fdf0e6']
for (y, t), c in zip(boxes, cols):
    ax0.add_patch(FancyBboxPatch((0.06, y - 0.08), 0.88, 0.16, boxstyle='round,pad=0.01', fc=c, ec='#5e5e5a', lw=0.6))
    ax0.text(0.5, y + 0.005, t, ha='center', va='center', fontsize=7.6)
for y in (0.865, 0.622, 0.378):
    ax0.annotate('', xy=(0.5, y - 0.153), xytext=(0.5, y - 0.09), arrowprops=dict(arrowstyle='-|>', color='#5e5e5a', lw=0.9, mutation_scale=9))
ax = fig.add_axes([0.49, 0.17, 0.48, 0.62])
d = np.load('out/cur_seq_300_1.npz'); tw = d['tw']; Tn = len(tw) - 1; Td = 300
S = len(d['static']) // (Tn + 1)
for k, c, lab in (('static', '#2a78d6', 'Static compensation'), ('prop_hist', '#1baf7a', 'Proposed (history-aware)')):
    for s in range(S):
        e = d[k][s * (Tn + 1):(s + 1) * (Tn + 1)]
        ax.semilogy((s * Td * 1e-6 + tw[1:]) * 1e3, np.clip(e[1:], 1e-6, None), color=c, lw=1.0, label=lab if s == 0 else None)
ax.axhline(1e-3, color='#5e5e5a', lw=0.6, ls=(0, (2, 2)), label='$10^{-3}$ error (about 10 bits)')
ax.set_ylim(1e-4, 2); ax.set_xlim(0, 1.8)
ax.set_xlabel('Time (ms), new matrix every 300 µs', fontsize=7.5); ax.set_ylabel('Matrix error', fontsize=7.5)
ax.tick_params(labelsize=7); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, handlelength=1.6, columnspacing=1.0)
fig.savefig('tex/graphical_abstract.png', dpi=300); fig.savefig('tex/graphical_abstract.pdf'); fig.savefig('tex/graphical_abstract.tif', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
