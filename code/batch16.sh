#!/bin/bash
mkdir -p out
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for s in "$@"; do python3 run16.py $s >> out/log_N16.txt 2>&1; done
