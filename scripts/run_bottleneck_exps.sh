#!/usr/bin/env bash

ray exec "python flow/examples/rllib/multiagent_exps" --start --stop --cluster-name=ev_test1 --tmux