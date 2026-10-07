"""Generate tex/tab_main.tex (N=8 and N=16 single reconfigurations) and tex/tab_seq.tex."""
import json, glob, math
import numpy as np
import os
os.makedirs('tex', exist_ok=True)
r0 = lambda x: 'never' if not math.isfinite(x) else str(int(math.floor(x + 0.5 + 1e-6)))
def load(tag): return sorted([json.load(open(f)) for f in glob.glob(f'out/res_{tag}_*.json')], key=lambda r: r['seed'])
def cell(R, k, key):
    v = np.array([r[k][key] for r in R]) * 1e6
    if not np.isfinite(v).all():
        return 'never'
    return f'{r0(np.median(v))} [{r0(v.min())}, {r0(v.max())}]'
lab = {'static': 'Static compensation', 'siso': 'Per-heater pre-emphasis', 'prop': 'Proposed'}
rows = []; naive_txt = []
for tag, N in (('main', 8), ('N16', 16)):
    R = load(tag)
    nv = [r['naive']['final_err'] for r in R]
    rows.append(f'\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{$N={N}$, {len(R)} reconfigurations}}}}\\\\')
    naive_txt.append(f'{np.median(nv):.3f} [{min(nv):.3f}, {max(nv):.3f}] for $N={N}$')
    for k in ('static', 'siso', 'prop'):
        rows.append(f'\\quad {lab[k]} & {cell(R, k, "ts_1e-2")} & {cell(R, k, "ts_1e-3")}\\\\')
tex = r"""\begin{table}[t]
\centering
\caption{Single Reconfigurations on the Three-Dimensional Model: Median [Range] Settling Time (\si{\micro s})}
\label{tab:main}
\setlength{\tabcolsep}{4pt}
\footnotesize
\begin{tabular}{@{}lcc@{}}
\toprule
Strategy & $t_{10^{-2}}$ & $t_{10^{-3}}$\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\\[2pt]{\footnotesize $t_\varepsilon$: time after which $e(t)\le\varepsilon$ within \SI{10}{ms}. Naive calibration never settles; its steady error is NAIVE.}
\end{table}
"""
tex = tex.replace('NAIVE', ' and '.join(naive_txt))
open('tex/tab_main.tex', 'w').write(tex)
print('\n'.join(rows))

s = json.load(open('out/seq_n16_summary.json'))
lab2 = {'static': 'Static compensation', 'siso': 'Per-heater pre-emphasis', 'prop_ss': 'Proposed, steady-start', 'prop_hist': 'Proposed, history-aware'}
rows = []
for Td, name in (('300', '\\SI{300}{\\micro s}'), ('1000', '\\SI{1}{ms}')):
    T = s[Td]
    rows.append(f'\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{Dwell time {name}}}}}\\\\')
    for k in ('static', 'siso', 'prop_ss', 'prop_hist'):
        c = []
        for e in ('1e-2', '1e-3'):
            x = T[k][e]
            med = '--' if x['median_us'] is None else r0(x['median_us'])
            c += [f"{x['n_settled']}/{x['n']}, {med}", f"{x['usable_pct']:.0f}"]
        rows.append(f'\\quad {lab2[k]} & ' + ' & '.join(c) + '\\\\')
tex = r"""\begin{table}[t]
\centering
\caption{Reprogramming Sequences ($N=8$): Windows That Settle, Median Settling Time (\si{\micro s}) and Usable Fraction of the Dwell Time (\%)}
\label{tab:seq}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{$\varepsilon=10^{-2}$} & \multicolumn{2}{c}{$\varepsilon=10^{-3}$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Strategy & settled, $t_\varepsilon$ & usable & settled, $t_\varepsilon$ & usable\\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\\[2pt]{\footnotesize Windows 2 to 6 of three sequences of six random configurations (15 windows per row), all starting before the previous heat has settled. Median shown when more than half of the windows settle. Usable: mean of $1-t_\varepsilon/T_{\mathrm d}$, zero for windows that do not settle.}
\end{table}
"""
open('tex/tab_seq.tex', 'w').write(tex)
print('\n'.join(rows))
