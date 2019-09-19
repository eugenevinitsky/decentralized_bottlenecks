#!/usr/bin/env bash

# This runs on a policy on a cluster and sync the output graphs to s3

startdate="09-15-2019"

ray up ray_autoscale.yaml --cluster-name br_plot -y

ray exec ray_autoscale.yaml "aws s3 sync s3://eugene.experiments/trb_bottleneck_paper/ ./ ; \
\
python flow/flow/visualize/bottleneck_results.py ~/$startdate/\
MA_NLC_NCM_NLSTM_AG_NCN_CVF/MA_NLC_NCM_NLSTM_AG_NCN_CVF/CCPPOTrainer_MultiBottleneckEnv-v0_0_lr=5e-05_2019-09-15_21-56-18m90fb4uj 350 MA_NLC_NCM_NLSTM_AG_NCN_CVF \
--num_trials 1 --outflow_min 400 --outflow_max 500  --cluster_mode ; \
\
mkdir -p ~/flow/flow/visualize/trb_data/av_results/$startdate ; \
\
mv ~/flow/flow/visualize/trb_data/av_results/tmp ~/flow/flow/visualize/trb_data/av_results/$startdate/$startdate ; \
\
aws s3 sync ~/flow/flow/visualize/trb_data/av_results/$startdate/ s3://eugene.experiments/trb_bottleneck_paper/policy_graphs" --cluster-name br_plot --stop
