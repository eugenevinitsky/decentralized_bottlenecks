#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/11-11-2019/centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR/centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR/PPO_DesiredVelocityEnv-v0_1_max_seq_len=40,num_sgd_iter=10_2019-11-11_18-59-39cpi296mk 400 t\
est --num_trials 3 --outflow_min 2600 --outflow_max 2600 --num_cpus 1 --render_mode sumo_gui
