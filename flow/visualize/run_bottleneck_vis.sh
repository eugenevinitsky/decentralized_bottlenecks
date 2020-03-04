#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/03-02-2020/2pen_i1900/2pen_i1900/PPO_MultiBottleneckEnv-v0_3d7882a6_2020-03-03_05-24-22nt39l_bx 450 test \
--num_trials 2 --outflow_min 1900 --outflow_max 1900 --num_cpus 1 --render_mode sumo_gui --local_mode
