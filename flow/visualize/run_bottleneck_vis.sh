#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/03-22-2020/i2400_td3senv_ncrit12/i2400_td3senv_ncrit12/TD3_6_actor_lr=0.001,critic_lr=0.0001,n_step=5,prioritized_replay=True_2020-03-22_17-27-48mi8rif4a 100 test \
--num_trials 2 --outflow_min 2400 --outflow_max 2400 --num_cpus 1 --render_mode sumo_gui --local_mode
