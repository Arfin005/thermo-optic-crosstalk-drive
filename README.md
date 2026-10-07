# Cancelling Transient Thermal Crosstalk in Thermo-Optic Photonic Matrix Processors

Code and result data for the paper

> Md. Al Mohaimin Bhuiyan Arfin, "Cancelling Transient Thermal Crosstalk for Fast Reprogramming of
> Thermo-Optic Photonic Matrix Processors," submitted to *IEEE Journal of Selected Topics in Quantum
> Electronics*, Special Issue on Emerging Photonic Computing.

The repository contains a three-dimensional finite-volume thermal model of 8×8 and 16×16
silicon-on-insulator Clements meshes (solved exactly by modal separation), identification of a compact
shared-pole thermal model, gradient-based design of heater drive waveforms (single reconfigurations and
history-aware design for reprogramming sequences), and the evaluation of all strategies on the
three-dimensional model.

## Requirements
Python 3.10 or later and the packages in `requirements.txt`:

    pip install -r requirements.txt

## Reproducing the results (run from `code/`)
| Step | Command | Output |
|---|---|---|
| 1 | `python fd_validate.py` | modal solution against the direct solver (Section II-C) |
| 2 | `python fd_char.py` | heater step responses, P_pi, crosstalk, grid convergence (`fd_char.json`) |
| 3 | `python ident.py` | step responses of all 56 heaters of the 8×8 mesh (`ref_steps.npy`, `ref_dc.npy`) |
| 4 | `python tab_rom.py` | compact-model accuracy against order (Section IV-A) |
| 5 | `./batch3.sh main 24 8 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | 8×8 single reconfigurations (Table II) |
| 6 | `python ident16.py` | step responses and compact model of the 16×16 mesh (`R16.npy`, about 11 min) |
| 7 | `./batch16.sh 1 2 3 4 5 6` | 16×16 single reconfigurations (Table II, about 30 min each) |
| 8 | `./seqbatch.sh` | reprogramming sequences, 300 µs and 1 ms dwell (Table III, Fig. 5) |
| 9 | `./queue.sh` | model-order and driver-limit studies (Table IV) |
| 10 | `python variation.py 1 2 3 4 5 6` | fabrication variations (Table IV) |
| 11 | `python quant.py 1 2 3 4 5 6` | DAC resolution (Table IV) |
| 12 | `python fig_model.py; python fig_results.py; python fig_seq.py; python fig_ga.py` | figures, graphical abstract and summary (`out/seq_n16_summary.json`) |
| 13 | `python tab_jstqe.py; python tab_robust.py` | Tables II, III and IV |

Scripts write their output to `code/out/` (data) and `code/tex/` (tables and figures). An 8×8
reconfiguration takes about 2 to 4 minutes on one CPU core. `ident_fast.py` computes exact step
responses of all heaters by contracting the separable modal solution, which makes the 16×16 mesh
practical. `qp.py` is the convex quadratic-programming variant mentioned, and not adopted, in
Section IV-C of the paper.

## Results
`results/` holds the data behind every number in the paper: `res_<study>_<seed>.json` (settling
times and integrated errors of every strategy; `main` for 8×8, `N16` for 16×16, `seq_<dwell>` for
sequences), `seq_n16_summary.json`, `ident16.json`, `variation.json`, `quant.json`, `tab_rom.json`,
`fd_char.json` and `ident_fit.json`, and the error curves of the two sequences shown in Fig. 5
(`cur_seq_300_1.npz`, `cur_seq_1000_1.npz`).

## Licence
MIT (see `LICENSE`).
