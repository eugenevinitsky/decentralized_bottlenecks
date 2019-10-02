#!/bin/bash

python bottleneck_results.py \
/Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/09-28-2019/high_in_NCN_AGG_NLSTM_1pen_CVF/high_in_NCN_AGG_NLSTM_1pen_CVF/CCPPOTrainer_MultiBottleneckEnv-v0_2_lr=5e-05_2019-09-28_22-30-08agblphm5 1000 t\
est --num_trials 2 --outflow_min 2300 --outflow_max 2330 --num_cpus 1 # --render_mode sumo_gui
