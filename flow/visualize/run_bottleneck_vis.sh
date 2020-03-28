#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/03-28-2020/l2400_h2400_td3_av0p4_senv_out_buff5e5/l2400_h2400_td3_av0p4_senv_out_buff5e5/TD3_6_actor_lr=0.001,critic_lr=0.0001,n_step=5,prioritized_replay=True_2020-03-28_20-03-512_imx772 100 test \
--num_trials 1 --outflow_min 2400 --outflow_max 2400 --num_cpus 1 --render_mode sumo_gui --local_mode
