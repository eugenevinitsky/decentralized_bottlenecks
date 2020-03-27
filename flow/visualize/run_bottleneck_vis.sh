#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/03-27-2020/l2400_h2400_td3_ncrit8_av0p1_senv_out/l2400_h2400_td3_ncrit8_av0p1_senv_out/TD3_6_actor_lr=0.001,critic_lr=0.0001,n_step=5,prioritized_replay=True_2020-03-27_17-17-33gu8_aetr 100 test \
--num_trials 2 --outflow_min 2400 --outflow_max 2400 --num_cpus 1 --render_mode sumo_gui --local_mode
