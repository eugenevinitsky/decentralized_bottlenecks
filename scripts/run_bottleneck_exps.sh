#!/usr/bin/env bash

ray exec ray_autoscale.yaml "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MultiDecentralObsBottleneckCurric\
 --num_cpus 14 --use_s3 \
--grid_search --multi_node --rollout_scale_factor 2 --num_samples 2 --curriculum --num_curr_iters 150" \
--start --stop --cluster-name=ev_test1 --tmux

ray exec ray_autoscale.yaml "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MultiCentralObsBottleneckCurric\
 --num_cpus 5 --use_s3 \
--grid_search --multi_node --central_obs --rollout_scale_factor 2 --num_samples 2 --curriculum --num_curr_iters 150" \
--start --stop --cluster-name=ev_test2 --tmux

ray exec ray_autoscale.yaml "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MultiDecentralObsBottleneckCurric3400\
 --num_cpus 14 --use_s3 \
--grid_search --multi_node --rollout_scale_factor 2 --num_samples 2 --low_inflow 3400 --high_inflow 3400 --curriculum --num_curr_iters 150" \
--start --stop --cluster-name=ev_test3 --tmux

ray exec ray_autoscale.yaml "python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MultiCentralObsBottleneckCurric3400\
 --num_cpus 5 --use_s3 \
--grid_search --multi_node --central_obs --rollout_scale_factor 2 --num_samples 2 --low_inflow 3400 --high_inflow 3400 --num_curr_iters 150" \
--start --stop --cluster-name=ev_test4 --tmux