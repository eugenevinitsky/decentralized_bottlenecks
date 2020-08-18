#!/bin/bash

# download graphs data
aws s3 sync s3://nathan.experiments/trb_bottleneck_paper/graphs_data ./data/
# including alinea_vs_controller (baseline)
aws s3 sync s3://nathan.experiments/trb_bottleneck_paper/alinea_vs_controller ./data/alinea_vs_controller



# the following is to generate all graphs on AWS for the following experiments
# already generated, no need to rerun

# simple env - no aggregate - 5% penetration
# simple env - no aggregate - 10% penetration
# simple env - no aggregate - 20% penetration
# simple env - no aggregate - 40% penetration
# simple env - with aggregate - 5% penetration
# simple env - with aggregate - 10% penetration
# simple env - with aggregate - 20% penetration
# simple env - with aggregate - 40% penetration
# complex env - with aggregate - 5% penetration
# complex env - with aggregate - 10% penetration
# complex env - with aggregate - 20% penetration
# complex env - with aggregate - 40% penetration


if false; then
    for exp in 07-26-2020/seedsearch_dqzj2_0p05/seedsearch_dqzj2_0p05/TD3_25_seed=24_2020-07-26_20-03-240trecw9h \
            07-26-2020/seedsearch_dqzj2_0p1/seedsearch_dqzj2_0p1/TD3_30_seed=29_2020-07-26_20-07-11oe2hv4gz \
            07-26-2020/seedsearch_dqzj2_0p2/seedsearch_dqzj2_0p2/TD3_16_seed=15_2020-07-26_20-10-47m7h7gge_ \
            07-26-2020/seedsearch_dqzj2_0p4/seedsearch_dqzj2_0p4/TD3_10_seed=9_2020-07-26_20-14-55k__w7kmt \
            08-14-2020/seedsearch_pos09_0p05/seedsearch_pos09_0p05/TD3_29_seed=28_2020-08-14_08-39-16tmtj4mkr \
            08-14-2020/seedsearch_pos09_0p1/seedsearch_pos09_0p1/TD3_4_seed=3_2020-08-14_08-44-22pa20pu0_ \
            08-14-2020/seedsearch_pos09_0p2/seedsearch_pos09_0p2/TD3_0_seed=None_2020-08-14_08-49-0145nytugu \
            08-14-2020/seedsearch_pos09_0p4/seedsearch_pos09_0p4/TD3_10_seed=9_2020-08-14_08-53-25o0bkk0id \
            07-26-2020/seedsearch_o9di2_0p05/seedsearch_o9di2_0p05/TD3_15_seed=14_2020-07-26_19-50-079dxaty9f \
            07-26-2020/seedsearch_o9di2_0p1/seedsearch_o9di2_0p1/TD3_17_seed=16_2020-07-26_19-53-16cx7riof4 \
            07-26-2020/seedsearch_o9di2_0p2/seedsearch_o9di2_0p2/TD3_18_seed=17_2020-07-26_19-56-22fcx3lomy \
            07-26-2020/seedsearch_o9di2_0p4/seedsearch_o9di2_0p4/TD3_20_seed=19_2020-07-26_19-59-47qydk6h9_
    do
        for pen in 0.05 0.1 0.2 0.4
        do
            echo ${exp} ${pen}
            ray exec ray_autoscale.yaml \
            "python flow/flow/visualize/generate_graphs.py ${exp} 2000 ${pen}" \
            --start --stop --tmux --cluster-name nathan_graphs_${pen}_$(od -N 4 -t uL -An /dev/urandom | tr -d " ") &
        done
    done
fi
