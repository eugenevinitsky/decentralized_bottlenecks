#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/ray_results/test/CCPPOTrainer_MultiBottleneckEnv-v0_4e50a3c4_2020-03-07_17-44-10dumk6ku7 20 test \
--num_trials 2 --outflow_min 2400 --outflow_max 2400 --num_cpus 1 --render_mode sumo_gui --local_mode
