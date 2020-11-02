<img src="docs/img/square_logo.png" align="right" width="25%"/>

[![Build Status](https://travis-ci.com/flow-project/flow.svg?branch=master)](https://travis-ci.com/flow-project/flow)
[![Docs](https://readthedocs.org/projects/flow/badge)](http://flow.readthedocs.org/en/latest/)
[![Coverage Status](https://coveralls.io/repos/github/flow-project/flow/badge.svg?branch=master)](https://coveralls.io/github/flow-project/flow?branch=master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flow-project/flow/binder)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flow-project/flow/blob/master/LICENSE.md)

# Flow

This is a fork of [Flow](https://flow-project.github.io/), a computational framework for deep RL and control experiments for traffic microsimulation. Please refer to the [original repository](https://github.com/flow-project/flow/blob/master) for installation instructions. 
This fork is used to provide an implementation for [1]. 

# Running experiments

This section present how to run the experiments used to generate the paper, and visualize the results.

## Training

Run the following command to train for 2000 iterations at an AV penetration rate of 10% (`--av_frac`) on the minimal state space. 

```script
python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py experiment_name --num_iters 2000 --checkpoint_freq 100 --av_frac 0.1 \
--num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 \
--sim_step 0.5 --sims_per_step 5 --reroute_on_exit --td3 --grid_search \
--simple_env
```

To train on the radar state space, simply remove the `--simple_env` flag. To add aggregate observations to the state space, add the `--aggregate_info` flag. To train a universal controller with a penetration randomly chosen between 5% and 40% at the beginning of each episode, use a negative penetration rate such as `--av_frac -1`. 

Optionally, add the `--lc_on` tag to enable lane changing among human vehicles, or `--no_congest_number` to remove bottleneck information from the state space. 

## Visualization

To render a trained policy, use Flow's visualizer script for RLlib. Example usage:

```script
python flow/visualize/visualizer_rllib.py ~/ray_results/path/to/checkpoint 2000
```

where 2000 is the checkpoint number.

## Graph generation

We generated the graph data (evaluation of the policy at different inflows) using 

```script
python flow/visualize/generate_graphs.py <experiment_checkpoint_path> <checkpoint_number> [<evaluation penetration>]
```

which works with our AWS S3 storage. To generate the data locally, see `flow/visualize/bottleneck_results.py` directly. To then generate the graphs from that data, see `generate_graphs/generate_graphs.py` from which you can generate graphs from your own data by adaptain the `__main__` section. 

# References

[1] Vinitsky, Lichtlé, Parvate, Bayen, "Optimizing Mixed Autonomy Traffic Flow With Decentralized Autonomous Vehicles and Multi-Agent RL." arXiv preprint TODO add link (2020)
