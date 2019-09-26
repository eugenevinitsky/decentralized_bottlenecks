#!/bin/bash

python bottleneck_results.py \
/Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/09-25-2019/high_in_NCN_AGG_NLSTM_CVF/high_in_NCN_AGG_NLSTM_CVF/CCPPOTrainer_MultiBottleneckEnv-v0_0_lr=5e-05_2019-09-25_20-08-36pzku_j6k 150 t\
est --num_trials 2 --outflow_min 2300 --outflow_max 2300 --num_cpus 1 --render_mode sumo_gui
