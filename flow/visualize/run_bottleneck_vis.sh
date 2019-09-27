#!/bin/bash

python bottleneck_results.py \
/Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/09-26-2019/high_in_NCN_AGG_NLSTM_Past_CVF/high_in_NCN_AGG_NLSTM_Past_CVF/CCPPOTrainer_MultiBottleneckEnv-v0_0_lr=5e-05_2019-09-27_00-34-27xw716to8 150 t\
est --num_trials 2 --outflow_min 2300 --outflow_max 2330 --num_cpus 1 --render_mode sumo_gui
