"""Figure and table data for the reprogramming-sequence study and the 16x16 study."""
import glob, json, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = {'static': '#2a78d6', 'siso': '#eb6834', 'prop_ss': '#9b59b6', 'prop_hist': '#1baf7a'}
LS = {'static': '-', 'siso': '--', 'prop_ss': '-.', 'prop_hist': '-'}
LAB = {'static': 'Static compensation', 'siso': 'Per-heater pre-emphasis',
       'prop_ss': 'Proposed, steady-start designs', 'prop_hist': 'Proposed, history-aware'}
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.5, 'lines.linewidth': 1.1,
                     'savefig.dpi': 300})
out = 'tex/figs'
os.makedirs(out, exist_ok=True)

fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6))
for ax, Td, lab in zip(axs, (300, 1000), ('(a)', '(b)')):
    d = np.load(f'out/cur_seq_{Td}_1.npz')
    tw = d['tw']; Tn = len(tw) - 1; S = len(d['static']) // (Tn + 1)
    for k in ('static', 'prop_ss', 'prop_hist'):
        for s in range(S):
            e = d[k][s * (Tn + 1):(s + 1) * (Tn + 1)]
            ax.semilogy((s * Td * 1e-6 + tw[1:]) * 1e3, np.clip(e[1:], 1e-6, None), color=C[k], ls=LS[k],
                        label=LAB[k] if s == 0 else None)
    for s in range(1, S):
        ax.axvline(s * Td * 1e-3, color='#c8c8c4', lw=0.5, zorder=0)
    for eps in (1e-2, 1e-3):
        ax.axhline(eps, color='#5e5e5a', lw=0.6, ls=(0, (2, 2)))
    ax.set_ylim(1e-4, 2); ax.set_xlim(0, S * Td * 1e-3)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Matrix error $e(t)$')
    ax.set_title(f'{lab} Dwell time {Td} µs' if Td < 1000 else f'{lab} Dwell time 1 ms', loc='left', fontsize=8)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(f'{out}/fig_seq.pdf'); plt.close(fig)

# ---- table data
def summary(Td):
    R = [json.load(open(f)) for f in sorted(glob.glob(f'out/res_seq_{Td}_*.json'))]
    T = {}
    for k in ('static', 'siso', 'prop_ss', 'prop_hist'):
        row = {}
        for eps in ('1e-2', '1e-3'):
            v = np.array([x for r in R for x in r[k]['ts_' + eps][1:]])   # windows s >= 2 (history)
            fin = np.isfinite(v)
            use = np.where(fin, (Td * 1e-6 - np.where(fin, v, 0)) / (Td * 1e-6), 0.0)
            row[eps] = {'n_settled': int(fin.sum()), 'n': int(len(v)),
                        'median_us': float(np.median(v) * 1e6) if fin.sum() * 2 > len(v) else None,
                        'usable_pct': float(use.mean() * 100)}
        v1 = np.array([r[k]['ts_1e-3'][0] for r in R])
        row['first_window_ts_1e-3_us'] = [float(x * 1e6) for x in v1]
        row['e_end_median'] = float(np.median([x for r in R for x in r[k]['e_end'][1:]]))
        T[k] = row
    T['design_s_per_window'] = float(np.mean([r['design_s'] / (2 * r['S']) for r in R]))
    T['n_seq'] = len(R)
    return T

summ = {Td: summary(Td) for Td in (300, 1000) if glob.glob(f'out/res_seq_{Td}_*.json')}
R16 = [json.load(open(f)) for f in sorted(glob.glob('out/res_N16_*.json'))]
if R16:
    s16 = {'n': len(R16)}
    for k in ('static', 'siso', 'prop'):
        s16[k] = {key: [float(np.median([r[k][key] for r in R16]) * (1e6 if key.startswith('ts') else 1)),
                        float(min(r[k][key] for r in R16) * (1e6 if key.startswith('ts') else 1)),
                        float(max(r[k][key] for r in R16) * (1e6 if key.startswith('ts') else 1))]
                  for key in ('ts_1e-2', 'ts_1e-3', 'iae_us')}
    s16['naive_final'] = [float(np.median([r['naive']['final_err'] for r in R16])), float(min(r['naive']['final_err'] for r in R16)), float(max(r['naive']['final_err'] for r in R16))]
    s16['ratio_1e-2'] = sorted(r['static']['ts_1e-2'] / r['prop']['ts_1e-2'] for r in R16)
    s16['ratio_1e-3'] = sorted(r['static']['ts_1e-3'] / r['prop']['ts_1e-3'] for r in R16)
    s16['faster_both'] = int(sum(r['prop']['ts_1e-2'] < r['static']['ts_1e-2'] and r['prop']['ts_1e-3'] < r['static']['ts_1e-3'] for r in R16))
    s16['design_s'] = [float(np.median([r['design_s'] for r in R16])), float(min(r['design_s'] for r in R16)), float(max(r['design_s'] for r in R16))]
    s16['total_power_W'] = float(np.median([x for r in R16 for x in (r['total_power_A_mW'], r['total_power_B_mW'])]) / 1e3)
    s16['VB_max'] = float(max(r['VB_max'] for r in R16)); s16['PB_max_mW'] = float(max(r['PB_max_mW'] for r in R16))
    s16['model_vs_3d'] = float(max(abs(r['prop_model'][k] - r['prop'][k]) for r in R16 for k in ('ts_1e-2', 'ts_1e-3')) * 1e6)
    s16['ident'] = json.load(open('out/ident16.json'))
    summ['N16'] = s16
json.dump(summ, open('out/seq_n16_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
