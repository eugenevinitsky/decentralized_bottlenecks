#!/usr/bin/env bash

# This runs on a policy on a cluster and sync the output graphs to s3

startdate="09-15-2019"

ray up ray_autoscale.yaml --cluster-name br_plot -y

ray exec ray_autoscale.yaml "aws s3 sync s3://eugene.experiments/trb_bottleneck_paper/ ./" --cluster-name br_plot --tmux

ray exec ray_autoscale.yaml "python flow/flow/visualize/bottleneck_results.py ~/$startdate/\
MA_NLC_NCM_NLSTM_AG_NCN_CVF/MA_NLC_NCM_NLSTM_AG_NCN_CVF/CCPPOTrainer_MultiBottleneckEnv-v0_0_lr=5e-05_2019-09-15_21-56-18m90fb4uj 350 test \
--num_trials 10 --outflow_min 400 --outflow_max 500" --cluster-name br_plot --tmux

# format the data and upload
d=$(date +%m-%d-%Y)

ray exec ray_autoscale.yaml "mkdir -p ~/flow/flow/visualize/trb_data/av_results/$startdate" --cluster-name br_plot --tmux
ray exec ray_autoscale.yaml "mv -p ~/flow/flow/visualize/trb_data/av_results/$d ~/flow/flow/visualize/trb_data/av_results/$startdate/$startdate" --cluster-name br_plot --tmux
ray exec "aws s3 sync ~/flow/flow/visualize/trb_data/av_results/$startdate/ s3://eugene.experiments/trb_bottleneck_paper/policy_graphs" --cluster-name br_plot --tmux
ray down ray_autoscale.yaml --cluster-name br_plot