#!/usr/bin/env bash

# This trains a qmix agent on the cluster

ray exec scripts/ray_autoscale.yaml \
"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py ma_kp_qmix_2500 --multi_node --qmix --num_iters 1000 --av_frac 0.1 --low_inflow 2500 --high_inflow 2500 --use_s3" --start --stop --tmux --cluster-name kp_ma_qmix

