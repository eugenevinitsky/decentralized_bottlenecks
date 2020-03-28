#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/03-27-2020/l2400_h2400_td3_ncrit12_av0p4_senv_out_buff5e5/l2400_h2400_td3_ncrit12_av0p4_senv_out_buff5e5/TD3_8_actor_lr=0.001,critic_lr=0.001,n_step=1,prioritized_replay=False_2020-03-28_00-55-42c76842cm 200 test \
--num_trials 2 --outflow_min 2400 --outflow_max 2400 --num_cpus 1 --render_mode sumo_gui --local_mode
