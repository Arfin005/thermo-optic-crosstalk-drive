#!/bin/bash
# usage: batch3.sh tag K Vmax seeds...
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p out
tag=$1; K=$2; V=$3; shift 3
for s in "$@"; do python3 run3.py $s $tag $K $V 5 >> out/log_$tag.txt 2>&1; done
