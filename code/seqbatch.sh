#!/bin/bash
mkdir -p out
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for Td in 300 1000; do for s in 1 2 3; do python3 seq.py $s $Td 6 >> out/log_seq.txt 2>&1; done; done
