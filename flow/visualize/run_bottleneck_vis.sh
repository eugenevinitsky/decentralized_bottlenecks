#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/11-13-2019/centralized_0pen_0p6_GRU_PPO_ns0p5_FAIR_PEN30/centralized_0pen_0p6_GRU_PPO_ns0p5_FAIR_PEN30/PPO_DesiredVelocityEnv-v0_4_max_seq_len=20,num_sgd_iter=10_2019-11-14_04-55-41v4557mup 400 t\
est --num_trials 3 --outflow_min 2600 --outflow_max 2600 --num_cpus 1 --render_mode sumo_gui
