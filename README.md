<img src="docs/img/square_logo.png" align="right" width="25%"/>

[![Build Status](https://travis-ci.com/flow-project/flow.svg?branch=master)](https://travis-ci.com/flow-project/flow)
[![Docs](https://readthedocs.org/projects/flow/badge)](http://flow.readthedocs.org/en/latest/)
[![Coverage Status](https://coveralls.io/repos/github/flow-project/flow/badge.svg?branch=master)](https://coveralls.io/github/flow-project/flow?branch=master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/flow-project/flow/binder)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/flow-project/flow/blob/master/LICENSE.md)

# Flow

This is a fork of [Flow](https://flow-project.github.io/), a computational framework for deep RL and control experiments for traffic microsimulation.
This fork is used to support the paper XXXXXX.

To run the experiments used to generate the paper, follow the instructions to set-up Flow and then run the script 
`./scripts/run_bottleneck_exps.sh` to launch a set of AWS scripts. 

Alternately, you can generate all the graphs from the paper and process them by running XXXX.

## todo

branches: 
- old_working_commit (TODO rename into master?) -> training, visualize exps
- generate_graphs -> generate graphs for paper

```
python flow/examples/rllib/multiagent_exps/multiagent_bottleneck.py expname --num_iters 2000 --checkpoint_freq 200 --av_frac 0.1 \
    --num_samples 1 --rollout_scale_factor 1.0 --horizon 400 --low_inflow 2400 --high_inflow 2400 --simple_env --aggregate_info \
    --sim_step 0.5 --sims_per_step 5 --reroute_on_exit \
    --grid_search --use_s3 --td3
```

