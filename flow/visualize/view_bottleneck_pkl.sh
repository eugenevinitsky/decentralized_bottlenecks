#!/usr/bin/env bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/old_bottleneck_tests/03-16-2020/MultiDecentralObsBottleneck/MultiDecentralObsBottleneck/PPO_MultiBottleneckEnv-v0_87bc1c52_2020-03-16_18-28-18w0zty2he 50 test \
--num_trials 2 --outflow_min 2400 --outflow_max 2400 --num_cpus 1 --render_mode sumo_gui --local_mode
