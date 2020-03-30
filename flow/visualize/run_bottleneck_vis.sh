#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/ray_results/i2400_td3senv/TD3_0_2020-03-29_15-48-50fhksghrm 100 test \
--num_trials 1 --outflow_min 2400 --outflow_max 2400 --num_cpus 1 --render_mode sumo_gui --local_mode
