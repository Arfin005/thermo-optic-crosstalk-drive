#!/bin/bash
# wait for the main batch, then model-order and headroom studies (two lanes)
while pgrep -f "batch3.sh main" > /dev/null; do sleep 20; done
(./batch3.sh K8 8 8 1 2 3 4 5 6; ./batch3.sh V65 24 6.5 1 2 3 4 5 6) &
(./batch3.sh K12 12 8 1 2 3 4 5 6; ./batch3.sh V10 24 10 1 2 3 4 5 6) &
wait
