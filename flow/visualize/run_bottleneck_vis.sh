#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/12-10-2019/ma_qmix_test/ma_qmix_test/VariableQMIX_MultiBottleneckEnv-v0_1_lr=0.0005_2019-12-11_02-10-349edlm_rw 350 test \
--num_trials 3 --outflow_min 2300 --outflow_max 2300 --num_cpus 1 --render_mode sumo_gui --qmix
