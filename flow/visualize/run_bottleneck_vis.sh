#!/bin/bash

python flow/visualize/bottleneck_results.py /Users/kanaad/code/research/learning-traffic/imitation_results/high_in_NCN_AGG_NLSTM_Past_2pen_04frac_im/high_in_NCN_AGG_NLSTM_Past_2pen_04frac_im/ImitationPPOTrainer_MultiBottleneckImitationEnv-v0_0_lr=5e-05_2019-12-13_03-19-33rl4x64m3 150 test \
--num_trials 3 --outflow_min 2300 --outflow_max 2300 --num_cpus 1 --render_mode sumo_gui
