#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/03-28-2020/l2400_h2400_td3_av0p4_senv_out_25s_t2/l2400_h2400_td3_av0p4_senv_out_25s_t2/TD3_2_actor_lr=0.001,critic_lr=0.0001,gamma=0.99,prioritized_replay=True_2020-03-29_06-54-15rj6ipss0 250 test \
--num_trials 1 --outflow_min 2400 --outflow_max 2400 --num_cpus 1 --render_mode sumo_gui --local_mode
