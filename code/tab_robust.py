"""Generate tex/tab_robust.tex from the result files (half-up rounding)."""
import json, glob, math
import numpy as np
import os
os.makedirs('tex', exist_ok=True)
r0 = lambda x: int(math.floor(x + 0.5 + 1e-6)) if math.isfinite(x) else 'never'
def load(tag): return sorted([json.load(open(f)) for f in glob.glob(f'out/res_{tag}_*.json')], key=lambda r: r['seed'])
def row(R, keyf=lambda r, k: r[k]):
    m = lambda k, key: np.median([keyf(r, k)[key] for r in R]) * 1e6
    f3 = sum(keyf(r, 'prop')['ts_1e-3'] < keyf(r, 'static')['ts_1e-3'] for r in R)
    return [r0(m('static', 'ts_1e-2')), r0(m('static', 'ts_1e-3')), r0(m('prop', 'ts_1e-2')), r0(m('prop', 'ts_1e-3')), f'{f3}/{len(R)}']
rows = []
rows.append(('Nominal design and chip', row([r for r in load('main') if r['seed'] <= 6])))
rom = json.load(open('out/tab_rom.json'))
rows.append(('\\multicolumn{6}{@{}l}{\\emph{Compact-model order}}', None))
rows.append(('\\quad $K=12$ (\\SI{3.6}{mrad})', row(load('K12'))))
rows.append(('\\quad $K=8$ (\\SI{24}{mrad})', row(load('K8'))))
rows.append(('\\multicolumn{6}{@{}l}{\\emph{Driver voltage limit}}', None))
rows.append(('\\quad $V_{\\max}=\\SI{6.5}{V}$', row(load('V65'))))
rows.append(('\\quad $V_{\\max}=\\SI{10}{V}$', row(load('V10'))))
q = json.load(open('out/quant.json'))
rows.append(('\\multicolumn{6}{@{}l}{\\emph{DAC resolution} (static error floor)}', None))
for b in (12, 14, 16):
    R = [x for x in q if x['bits'] == b]
    fl = np.median([x['static_floor'] for x in R])
    rows.append((f'\\quad {b} bits (\\num{{{fl:.1e}}})', row(R)))
v = json.load(open('out/variation.json'))
names = {'k_ox -10%': 'Oxide conductivity $-10\\%$', 'k_ox +10%': 'Oxide conductivity $+10\\%$',
         't_box 1.8um': 'BOX \\SI{1.8}{\\micro m}', 't_box 2.2um': 'BOX \\SI{2.2}{\\micro m}',
         'z_h 2.2um': 'Heater height $+\\SI{0.5}{\\micro m}$', 'R0 -5%': 'Heater resistance $-5\\%$',
         'R0 +5%': 'Heater resistance $+5\\%$'}
rows.append(('\\multicolumn{6}{@{}l}{\\emph{Chip differs from design model}}', None))
for c, nm in names.items():
    R = [x for x in v if x['corner'] == c]
    rows.append(('\\quad ' + nm, row(R)))
body = '\n'.join(n + r' \\' if r is None else n + ' & ' + ' & '.join(str(x) for x in r) + r'\\' for n, r in rows)
tex = r"""\begin{table}[t]
\centering
\caption{Robustness ($N=8$): Median Settling Times (\si{\micro s}) of 6 Reconfigurations on the Three-Dimensional Model}
\label{tab:robust}
\setlength{\tabcolsep}{3pt}
\footnotesize
\begin{tabular}{@{}lccccc@{}}
\toprule
 & \multicolumn{2}{c}{Static} & \multicolumn{2}{c}{Proposed} & Faster at\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Case & $t_{10^{-2}}$ & $t_{10^{-3}}$ & $t_{10^{-2}}$ & $t_{10^{-3}}$ & $10^{-3}$\\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\\[2pt]{\footnotesize DAC: settling measured to the quantised steady state. Chip variations: each chip keeps its own static calibration and receives the correction designed on the nominal model.}
\end{table}
"""
open('tex/tab_robust.tex', 'w').write(tex)
print(body)
