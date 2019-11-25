#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/ray_results/test/ImitationPPOTrainer_MultiBottleneckImitationEnv-v0_0_2019-11-25_12-13-217osnogjc 80 test \
--num_trials 3 --outflow_min 2600 --outflow_max 2600 --num_cpus 1 --render_mode sumo_gui
