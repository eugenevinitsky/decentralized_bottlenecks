#!/bin/bash

python bottleneck_results.py \
/Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/09-19-2019/large_inflows_NCN/large_inflows_NCN/PPO_MultiBottleneckEnv-v0_2_lr=5e-05_2019-09-19_15-37-32an1akhg4  350 t\
est --num_trials 2 --outflow_min 2000 --outflow_max 2000 --num_cpus 1 # --render_mode sumo_gui
