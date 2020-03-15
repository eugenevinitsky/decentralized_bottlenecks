#!/usr/bin/env bash

ray exec ray_autoscale.yaml "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MultiDecentralObsBottleneck\
 --num_cpus 14 --use_s3 \
--grid_search --multi_node --rollout_scale_factor 2" \
--start --stop --cluster-name=ev_test1 --tmux

ray exec ray_autoscale.yaml "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MultiCentralObsBottleneck\
 --num_cpus 14 --use_s3 \
--grid_search --multi_node --central_obs --rollout_scale_factor 2" \
--start --stop --cluster-name=ev_test2 --tmux