#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/09-15-2019/\
MA_NLC_NCM_NLSTM_AG_NCN_CVF/MA_NLC_NCM_NLSTM_AG_NCN_CVF/CCPPOTrainer_MultiBottleneckEnv-v0_0_lr=5e-05_2019-09-15_21-56-18m90fb4uj 350 t\
est --num_trials 4 --outflow_min 400 --outflow_max 500 # --render_mode sumo_gui
