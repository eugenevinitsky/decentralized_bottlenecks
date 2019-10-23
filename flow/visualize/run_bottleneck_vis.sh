#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/10-20-2019/centralized_0pen_0p4_GRU_PPO_ns0p5_PEN/centralized_0pen_0p4_GRU_PPO_ns0p5_PEN/PPO_DesiredVelocityEnv-v0_0_lr=5e-05,max_seq_len=10,num_sgd_iter=10_2019-10-20_20-34-22_0zm9j81 300 t\
est --num_trials 2 --outflow_min 2600 --outflow_max 2600 --num_cpus 1 --render_mode sumo_gui
