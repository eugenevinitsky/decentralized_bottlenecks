#!/usr/bin/env bash

# 9/05/19 experiments
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
##    --num_samples 2 --grid_search --n_cpus 30 --use_s3 --rollout_scale_factor 0.5 --horizon 2000" \
##    --start --stop --cluster-name exp1 --tmux
##
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_LSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
##    --num_samples 2 --grid_search --n_cpus 30 --use_lstm --use_s3 --rollout_scale_factor 0.5 --horizon 2000" \
##    --start --stop --cluster-name exp2 --tmux
##
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_AG_NCN --num_iters 350 --checkpoint_freq 50 \
##    --num_samples 2 --grid_search --n_cpus 30 --use_s3 --aggregate_info --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_LSTM_AG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_lstm --use_s3 --aggregate_info --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp4 --tmux
#
#----------------------------------- Add communication ------------------------------------------------------------------------------------------
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_CM_NLSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_s3 --communicate --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_CM_LSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_lstm --use_s3 --communicate --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp6 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_CM_NLSTM_AG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_s3 --aggregate_info --communicate --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp7 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_CM_LSTM_AG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 30 --use_lstm --use_s3 --aggregate_info --communicate --rollout_scale_factor 0.5 --horizon 2000" \
#    --start --stop --cluster-name exp8 --tmux
####################################################################################################################################################
####################################################################################################################################################

# 9/10/19 experiments
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_CN_PEN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --congest_penalty" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_AG_NCN_PEN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --aggregate_info --rollout_scale_factor 1.0 --horizon 2000 --congest_penalty" \
#    --start --stop --cluster-name exp3 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/15/19 experiments with centralized vf
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py MA_NLC_NCM_NLSTM_NAG_CN_CVF --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --central_vf_size 64" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py MA_NLC_NCM_NLSTM_AG_NCN_CVF --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --aggregate_info --rollout_scale_factor 1.0 --horizon 2000 --central_vf_size 64" \
#    --start --stop --cluster-name exp3 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/19/19 experiments with centralized observations, 1 experiment with a longer timestep, and 1 experiment with a higher range of inflows
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_CN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --central_obs --high_inflow 2000" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py large_sim_step_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --sim_step 1.0 --high_inflow 2000" \
#    --start --stop --cluster-name exp2 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py large_inflows_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 1400 --high_inflow 2200" \
#    --start --stop --cluster-name exp3 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/20/19 experiments with a state space that contains edge position
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_CN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --central_obs --high_inflow 2000" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_NCN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --high_inflow 2000" \
#    --start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py MA_NLC_NCM_NLSTM_NAG_NCN_CVF --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --central_vf_size 64 --high_inflow 2000" \
#    --start --stop --cluster-name exp3 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/23/19 experiments with high inflows
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_NCN_high_in --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 1600 --high_inflow 2400" \
#    --start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py MA_NLC_NCM_NLSTM_NAG_NCN_high_in_PEN --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 1600 --high_inflow 2400 --congest_penalty" \
#    --start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py fixed_inflow --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --congest_penalty" \
#    --start --stop --cluster-name exp4 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py fixed_inflow_lstm --num_iters 350 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --congest_penalty --use_lstm" \
#    --start --stop --cluster-name exp5 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/25/19 experiments with fixed inflows and CVF and centralized obs and aggregate info

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_CN_NAGG_NLSTM --num_iters 200 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 \
#    --central_obs" \
#    --start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_CN_NAGG_LSTM --num_iters 200 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 4 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --use_lstm \
#    --central_obs" \
#    --start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_CN_NAGG_NLSTM_CVF --num_iters 200 --checkpoint_freq 50 --central_vf_size 64 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --central_obs" --start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_CN_NAGG_LSTM_CVF --num_iters 200 --checkpoint_freq 50 --central_vf_size 64 \
#--num_samples 2 --grid_search --n_cpus 4 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --use_lstm --central_obs" --start --stop --cluster-name exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_CN_AGG_NLSTM --num_iters 200 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info \
#    --central_obs" \
#    --start --stop --cluster-name exp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_CN_AGG_LSTM --num_iters 200 --checkpoint_freq 50 \
#    --num_samples 2 --grid_search --n_cpus 4 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --use_lstm --aggregate_info \
#    --central_obs" \
#    --start --stop --cluster-name exp6 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_CN_AGG_NLSTM_CVF --num_iters 200 --checkpoint_freq 50 --central_vf_size 64 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --central_obs" --start --stop --cluster-name exp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_CN_AGG_LSTM_CVF --num_iters 200 --checkpoint_freq 50 --central_vf_size 64 \
#--num_samples 2 --grid_search --n_cpus 4 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --use_lstm --aggregate_info --central_obs" --start --stop --cluster-name exp8 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 8 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_CVF --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --central_vf_size 64" \
#--start --stop --cluster-name exp9 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/26/19 experiments with an action history and varied amounts of staying in the bottleneck penalty

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_0LPen --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 0.0" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions" \
#--start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_0LPen_CVF --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 0.0 --central_vf_size 64" \
#--start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_Past_CVF --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --central_vf_size 64" \
#--start --stop --cluster-name exp4 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_0pen --num_iters 200 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 0.0" \
#--start --stop --cluster-name exp5 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/28/19 experiments with an action history and varied amounts of staying in the bottleneck penalty. Increased NN size.

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_Past_1pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 1.0 --central_vf_size 64" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_Past_2pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 2.0 --central_vf_size 64" \
#--start --stop --cluster-name exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_Past_3pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 3.0 --central_vf_size 64" \
#--start --stop --cluster-name exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_1pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 1.0 --central_vf_size 64" \
#--start --stop --cluster-name exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_2pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 2.0 --central_vf_size 64" \
#--start --stop --cluster-name exp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multi_bottleneck_centralized.py high_in_NCN_AGG_NLSTM_3pen_CVF --num_iters 1000 --checkpoint_freq 50 \
#--num_samples 2 --grid_search --n_cpus 2 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --life_penalty 3.0 --central_vf_size 64" \
#--start --stop --cluster-name exp6 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/29/19 experiments with 40% penetration rate

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_1pen_04frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 1.0" \
#--start --stop --cluster-name exp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_2pen_04frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 2.0" \
#--start --stop --cluster-name exp8 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_3pen_04frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.4 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 3.0" \
#--start --stop --cluster-name exp9 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_1pen_99frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.99 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 1.0" \
#--start --stop --cluster-name exp10 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_2pen_99frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.99 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 2.0" \
#--start --stop --cluster-name exp11 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py high_in_NCN_AGG_NLSTM_Past_3pen_9frac --num_iters 1000 --checkpoint_freq 50 --av_frac 0.99 \
#--num_samples 2 --grid_search --n_cpus 1 --use_s3 --rollout_scale_factor 1.0 --horizon 2000 --low_inflow 2300 --high_inflow 2301 --aggregate_info --keep_past_actions --life_penalty 3.0" \
#--start --stop --cluster-name exp12 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 9/30/19 experiments with centralized controller

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_LSTM --num_iters 400 --av_frac 0.1 \
#--num_samples 2 --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --life_penalty 0.0 --use_lstm" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_LSTM_PEN --num_iters 400 --av_frac 0.1 --congest_penalty \
#--num_samples 2 --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --life_penalty 0.0 --use_lstm" \
#--start --stop --cluster-name exp2 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 1/01/19 experiments with centralized controller. Added a grid search.

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_LSTM --num_iters 400 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --life_penalty 0.0 --use_lstm" \
#--start --stop --cluster-name exp1 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_LSTM_A3C --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2300 --high_inflow 2301 --life_penalty 0.0 --use_lstm --algorithm A3C" \
#--start --stop --cluster-name exp2 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/12/19 experiments with centralized controller over PPO, SAC, A3C w/ a GRU.

## 0.1 PEN
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C" \
#--start --stop --cluster-name exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name exp2 --tmux
#
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_SAC --num_iters 5000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
##--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC" \
##--start --stop --cluster-name exp3 --tmux
#
### 0.4 PEN
##
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_A3C --num_iters 3000 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C" \
#--start --stop --cluster-name exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO --num_iters 300 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name exp5 --tmux
##
##ray exec ray_autoscale.yaml \
##"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_SAC --num_iters 5000 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
##--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC" \
##--start --stop --cluster-name exp6 --tmux
##
### 1.0 PEN
##
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_GRU_A3C --num_iters 3000 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C" \
#--start --stop --cluster-name exp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_GRU_PPO --num_iters 300 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name exp8 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_SAC --num_iters 5000 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC" \
#--start --stop --cluster-name exp9 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/16/19 experiments with centralized controller over PPO, SAC, A3C w/ a GRU.

## 0.1 PEN
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name ev_exp2 --tmux
#
### 0.4 PEN
##
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_A3C --num_iters 3000 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500" \
#--start --stop --cluster-name ev_exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO --num_iters 300 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name ev_exp5 --tmux
#
### 1.0 PEN
##
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_GRU_A3C --num_iters 3000 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500" \
#--start --stop --cluster-name ev_exp7 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_1p0_GRU_PPO --num_iters 300 --av_frac 1.0 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 100 \
#--num_samples 1 --grid_search --n_cpus 36 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO" \
#--start --stop --cluster-name ev_exp8 --tmux

## 0.1 PEN
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C_ns20 --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 20 \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500 \
#--vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 20 \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C_ns0p5 --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500 \
#--vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp4 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_A3C_sr --num_iters 3000 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --speed_reward \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm A3C --checkpoint_freq 500 \
#--vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp5 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_sr --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --speed_reward \
#--num_samples 1 --grid_search --n_cpus 15 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 1200 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001" \
#--start --stop --cluster-name ev_exp6 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/20/19 experiments with centralized controller over PPO w/ a GRU and a congestion penalty
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_PEN --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--congest_penalty_start 30 --state_space_scaling 1 --congest_penalty" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_PEN --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--congest_penalty_start 20 --state_space_scaling 1 --congest_penalty" \
#--start --stop --cluster-name ev_exp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_PEN --num_iters 300 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--congest_penalty_start 30 --state_space_scaling 2 --congest_penalty" \
#--start --stop --cluster-name ev_exp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_PEN --num_iters 300 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--congest_penalty_start 20 --state_space_scaling 2 --congest_penalty" \
#--start --stop --cluster-name ev_exp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/22/19 experiments with the fair reward

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name ev_exp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name ev_exp2 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/27/19 experiments with the fair reward

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_bp25 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.25" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp25 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.25" \
#--start --stop --cluster-name evexp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_bp1 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1" \
#--start --stop --cluster-name evexp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp1 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1" \
#--start --stop --cluster-name evexp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/29/19 experiments with the fair reward

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_bp1_PEN20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp1_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_bp1_PEN20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_bp1_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --base_fair_reward 0.1 --congest_penalty  --congest_penalty_start 20" \
#--start --stop --cluster-name evexp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 10/30/19 experiments with the fair reward

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
#--start --stop --cluster-name evexp2 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp3 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty  --congest_penalty_start 30" \
#--start --stop --cluster-name evexp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 11/11/19 experiments with the fair reward. We have removed control outside of section 3 so the congest penalty is unnecessary.
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name evexp1 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_SAC_ns0p5_FAIR --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name evexp2 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name evexp3 --tmux

#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_SAC_ns0p5_FAIR --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 1 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --algorithm SAC --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60" \
#--start --stop --cluster-name evexp4 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 11/12/19 experiments with the fair reward. We have removed control outside of section 3 so the congest penalty is unnecessary.
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp1 --tmux
#
#ray exec ray_autoscale.yaml \
#"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
#--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
#--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
#--start --stop --cluster-name evexp2 --tmux

####################################################################################################################################################
####################################################################################################################################################
# 11/13/19 experiments with the paired fair reward. We have removed control outside of section 3 so the congest penalty is unnecessary.
ray exec ray_autoscale.yaml \
"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
--start --stop --cluster-name evexp1 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
--start --stop --cluster-name evexp2 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p6_GRU_PPO_ns0p5_FAIR_PEN30 --num_iters 400 --av_frac 0.6 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 30" \
--start --stop --cluster-name evexp3 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p1_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 300 --av_frac 0.1 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
--state_space_scaling 1 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
--start --stop --cluster-name evexp4 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p4_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 400 --av_frac 0.4 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
--start --stop --cluster-name evexp5 --tmux

ray exec ray_autoscale.yaml \
"python flow/examples/rllib/velocity_bottleneck.py centralized_0pen_0p6_GRU_PPO_ns0p5_FAIR_PEN20 --num_iters 400 --av_frac 0.6 --multi_node --sims_per_step 2 --sim_step 0.5 --warmup_steps 40 --num_sample_seconds 0.5 \
--num_samples 2 --grid_search --n_cpus 14 --use_s3 --rollout_scale_factor 2.0 --horizon 1000 --low_inflow 2000 --high_inflow 3000 --life_penalty 0.0 --use_gru --algorithm PPO --vf_loss_coeff .0001 \
--state_space_scaling 2 --fair_reward --exit_history_seconds 60 --congest_penalty --congest_penalty_start 20" \
--start --stop --cluster-name evexp6 --tmux
