#!/bin/bash

python bottleneck_results.py \
/Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/09-23-2019/fixed_inflow_lstm/fixed_inflow_lstm/PPO_MultiBottleneckEnv-v0_2_lr=5e-05,max_seq_len=10_2019-09-24_01-47-013_gynuzr 50 t\
est --num_trials 2 --outflow_min 2300 --outflow_max 2300 --num_cpus 1 --render_mode sumo_gui
