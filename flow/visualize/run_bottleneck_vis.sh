#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/10-16-2019/centralized_0pen_0p4_GRU_PPO/centralized_0pen_0p4_GRU_PPO/PPO_DesiredVelocityEnv-v0_0_lr=5e-05,max_seq_len=10,num_sgd_iter=10_2019-10-16_20-36-15ugkjal5s 50 t\
est --num_trials 2 --outflow_min 3000 --outflow_max 3000 --num_cpus 1 --render_mode sumo_gui
