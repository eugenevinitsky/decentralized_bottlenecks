#!/bin/bash

python bottleneck_results.py /Users/eugenevinitsky/Desktop/Research/Data/trb_bottleneck_paper/12-13-2019/3pen_im_200s/3pen_im_200s/ImitationPPOTrainer_MultiBottleneckImitationEnv-v0_1_lr=0.0005_2019-12-13_22-03-09reqzyuqo 300 test \
--num_trials 1 --outflow_min 1800 --outflow_max 1800 --num_cpus 1 --render_mode sumo_gui
