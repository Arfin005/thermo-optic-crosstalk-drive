# Crosstalk-Aware Feedforward Drive for Thermo-Optic Programmable Photonic Meshes

Code and result data for the paper

> Md. Al Mohaimin Bhuiyan Arfin, "Crosstalk-Aware Feedforward Drive for Fast Reconfiguration of
> Thermo-Optic Programmable Photonic Meshes," submitted to *IEEE Journal of Lightwave Technology*.

The repository contains a three-dimensional finite-volume thermal model of an 8×8 silicon-on-insulator
Clements mesh (solved exactly by modal separation), identification of a compact shared-pole thermal
model, gradient-based design of heater drive waveforms, and the evaluation of all strategies on the
three-dimensional model.

## Requirements
Python 3.10 or later and the packages in `requirements.txt`:

    pip install -r requirements.txt

## Reproducing the results (run from `code/`)
| Step | Command | Output |
|---|---|---|
| 1 | `python fd_validate.py` | modal solution against the direct solver |
| 2 | `python fd_char.py` | heater step responses, P_pi, crosstalk, grid convergence (`fd_char.json`) |
| 3 | `python ident.py` | step responses of all 56 heaters (`ref_steps.npy`, `ref_dc.npy`, about 5 min) |
| 4 | `python tab_rom.py` | compact-model accuracy against order (Table II) |
| 5 | `./batch3.sh main 24 8 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16` | main study, 16 reconfigurations (Table III) |
| 6 | `./queue.sh` | model-order and driver-limit studies (Table IV) |
| 7 | `python variation.py 1 2 3 4 5 6` | fabrication variations (Table V) |
| 8 | `python quant.py 1 2 3 4 5 6` | DAC resolution (Table IV) |
| 9 | `python fig_model.py; python fig_results.py` | figures |

Scripts write their output to `code/out/` (data) and `code/tex/` (tables and figures). Each reconfiguration takes about 2 to 4 minutes on one CPU core. `qp.py` is the convex
quadratic-programming variant described, and not adopted, in Section V-C of the paper.

## Results
`results/` holds the data behind every number in the paper: `res_<study>_<seed>.json` (settling
times and integrated errors of every strategy), `variation.json`, `quant.json`, `tab_rom.json`,
`fd_char.json` and `ident_fit.json`.

## Licence
MIT (see `LICENSE`).
