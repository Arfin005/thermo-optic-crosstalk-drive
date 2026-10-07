import glob, json, os, statistics as st
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = {'naive': '#8a8a85', 'static': '#2a78d6', 'siso': '#eb6834', 'prop': '#1baf7a'}
LS = {'naive': ':', 'static': '-', 'siso': '--', 'prop': '-'}
LAB = {'naive': 'Naive calibration', 'static': 'Static compensation',
       'siso': 'Per-heater pre-emphasis', 'prop': 'Proposed'}
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.5, 'lines.linewidth': 1.3,
                     'savefig.dpi': 300})
os.makedirs('tex/figs', exist_ok=True)


def load(tag):
    R = [json.load(open(f)) for f in glob.glob(f'out/res_{tag}_*.json')]
    return sorted(R, key=lambda r: r['seed'])


# ---------- example error curves
d = np.load('out/cur_main_1.npz'); t = d['t']
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for k in ('naive', 'static', 'siso', 'prop'):
    ax.loglog(t[1:] * 1e6, np.clip(d[k][1:], 1e-6, None), color=C[k], ls=LS[k], label=LAB[k])
for eps in (1e-2, 1e-3):
    ax.axhline(eps, color='#5e5e5a', lw=0.6, ls=(0, (2, 2)))
ax.set_xlim(0.25, 1e4); ax.set_ylim(1e-5, 3)
ax.set_xlabel('Time after reconfiguration command (µs)'); ax.set_ylabel('Matrix error $e(t)$')
ax.legend(frameon=False, fontsize=6.3, loc='lower left')
fig.tight_layout(); fig.savefig('tex/figs/fig_example.pdf'); plt.close(fig)

# ---------- example waveforms: (a) drive of two heaters, (b) the multi-heater correction
dV = np.abs(d['VB'] - d['VA']); sel = np.argsort(-dV)[:2]
tz = t[:-1] * 1e6
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.0), gridspec_kw={'width_ratios': [1, 1.25]})
ax = axs[0]
for h, a in zip(sel, (1.0, 0.55)):
    for k, V in (('static', np.repeat(d['VB'][None], len(tz), 0)), ('siso', d['Vs']), ('prop', d['Vl'])):
        ax.step(np.r_[-2, 0, tz[1:]], np.r_[d['VA'][h], V[0, h], V[1:, h]], where='post', color=C[k], ls=LS[k], alpha=a,
                label=LAB[k] if h == sel[0] else None)
ax.set_xlim(-2, 15); ax.set_ylim(0, 9.5); ax.set_xlabel('Time (µs)'); ax.set_ylabel('Heater voltage (V)')
ax.set_title('(a) Drive of two heaters', loc='left', fontsize=8)
ax.legend(frameon=False, fontsize=6.3, loc='lower right')
ax = axs[1]
corr = (d['Vl'] - d['Vs']) * 1e3
ton = np.array([tz[np.nonzero(d['Vs'][:, h] != d['Vs'][-1, h])[0][-1] + 1] if (d['Vs'][:, h] != d['Vs'][-1, h]).any() else 0.0 for h in range(corr.shape[1])])
mask = tz[:, None] > ton[None, :] + 0.5
cm = np.where(mask, corr, np.nan)
big = np.argsort(-np.nanmax(np.abs(np.where(tz[:, None] > 20, cm, np.nan)), 0))[:6]
for i, h in enumerate(big):
    ax.semilogx(tz, cm[:, h], lw=1.0, color=plt.cm.viridis(i / 6))
ax.axhline(0, color='#5e5e5a', lw=0.6)
ax.set_xlim(0.25, 3e3); ax.set_xlabel('Time (µs)'); ax.set_ylabel('Correction (mV)')
ax.set_title('(b) Multi-heater correction after the pre-emphasis pulse', loc='left', fontsize=8)
fig.tight_layout(); fig.savefig('tex/figs/fig_waves.pdf'); plt.close(fig)

# ---------- statistics over main reconfigurations
R = load('main')
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.1))
for ax, key, lab in zip(axs, ['ts_1e-2', 'ts_1e-3', 'iae_us'],
                        ['Settling to $10^{-2}$ (µs)', 'Settling to $10^{-3}$ (µs)', 'Integrated error (µs)']):
    for i, k in enumerate(('static', 'siso', 'prop')):
        v = np.array([r[k][key] for r in R]) * (1e6 if key.startswith('ts') else 1)
        # paired lines between strategies
        ax.scatter(i + np.linspace(-0.12, 0.12, len(v)), v, s=9, color=C[k], zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color='#1a1a19', lw=1.2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['Static', 'Per-heater', 'Proposed'])
    ax.set_title(lab, loc='left', fontsize=8); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig('tex/figs/fig_stats.pdf'); plt.close(fig)

# ---------- model order and headroom
def med(tag, k, key):
    R = load(tag)
    return (np.median([r[k][key] for r in R]) * 1e6, len(R)) if R else (np.nan, 0)
out = {}
if load('K8'):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.1))
    seeds6 = [1, 2, 3, 4, 5, 6]
    def sub(tag):
        return [r for r in load(tag) if r['seed'] in seeds6]
    for ax, key, lab in zip(axs, ['ts_1e-2', 'ts_1e-3'], ['Settling to $10^{-2}$ (µs)', 'Settling to $10^{-3}$ (µs)']):
        xs = [8, 12, 24]
        for k in ('static', 'siso', 'prop'):
            m = [np.median([r[k][key] for r in sub(tag)]) * 1e6 for tag in ('K8', 'K12', 'main')]
            ax.plot(xs, m, marker='o', ms=3.5, color=C[k], ls=LS[k], label=LAB[k])
        ax.set_xticks(xs); ax.set_xlabel('Compact-model order $K$'); ax.set_title(lab, loc='left', fontsize=8)
        ax.set_ylim(bottom=0)
    axs[1].legend(frameon=False, fontsize=6.3)
    fig.tight_layout(); fig.savefig('tex/figs/fig_order.pdf'); plt.close(fig)
if load('V65') and load('V10'):
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.1))
    seeds6 = [1, 2, 3, 4, 5, 6]
    for ax, key, lab in zip(axs, ['ts_1e-2', 'ts_1e-3'], ['Settling to $10^{-2}$ (µs)', 'Settling to $10^{-3}$ (µs)']):
        xs = [6.5, 8, 10]
        for k in ('static', 'siso', 'prop'):
            m = [np.median([r[k][key] for r in load(tag) if r['seed'] in seeds6]) * 1e6 for tag in ('V65', 'main', 'V10')]
            ax.plot(xs, m, marker='o', ms=3.5, color=C[k], ls=LS[k], label=LAB[k])
        ax.set_xticks(xs); ax.set_xlabel('Driver voltage limit $V_{\\max}$ (V)'); ax.set_title(lab, loc='left', fontsize=8)
        ax.set_ylim(bottom=0)
    axs[1].legend(frameon=False, fontsize=6.3)
    fig.tight_layout(); fig.savefig('tex/figs/fig_headroom.pdf'); plt.close(fig)
print('figures done')
