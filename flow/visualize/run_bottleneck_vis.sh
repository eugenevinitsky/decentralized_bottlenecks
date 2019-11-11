#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/10-30-2019/p20/centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN20/centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN20/PPO_DesiredVelocityEnv-v0_0_max_seq_len=20,num_sgd_iter=10_2019-10-30_20-13-27bcc0gqnz 300 t\
est --num_trials 3 --outflow_min 2600 --outflow_max 2600 --num_cpus 1 --render_mode sumo_gui
