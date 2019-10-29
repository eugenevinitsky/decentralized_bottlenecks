#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/10-27-2019/40_percent/centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp25/centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp25/PPO_DesiredVelocityEnv-v0_1_max_seq_len=40,num_sgd_iter=10_2019-10-28_06-37-29xb5cq0xe 400 t\
est --num_trials 3 --outflow_min 2600 --outflow_max 2600 --num_cpus 1 --render_mode sumo_gui
