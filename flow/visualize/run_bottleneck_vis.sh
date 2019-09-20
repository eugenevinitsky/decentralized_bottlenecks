#!/bin/bash

python bottleneck_results.py \
/Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/09-19-2019/MA_NLC_NCM_NLSTM_NAG_CN/MA_NLC_NCM_NLSTM_NAG_CN/PPO_MultiBottleneckEnv-v0_2_lr=5e-05_2019-09-19_15-35-225ywjmxdb 350 t\
est --num_trials 1 --outflow_min 400 --outflow_max 500 --num_cpus 4 #--render_mode sumo_gui
