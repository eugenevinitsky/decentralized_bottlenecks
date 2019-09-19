import argparse

def get_multiagent_bottleneck_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Parses command line args for multi-agent bottleneck exps')
    
    # required input parameters for tune
    parser.add_argument('exp_title', type=str, help='Informative experiment title to help distinguish results')
    parser.add_argument('--use_s3', action='store_true', help='If true, upload results to s3')
    parser.add_argument('--n_cpus', type=int, default=1, help='Number of cpus to run experiment with')
    parser.add_argument('--multi_node', action='store_true', help='Set to true if this will '
                                                                  'be run in cluster mode')
    parser.add_argument("--num_iters", type=int, default=350)
    parser.add_argument("--checkpoint_freq", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--grid_search", action='store_true')
    parser.add_argument('--rollout_scale_factor', type=float, default=1.0, help='the total number of rollouts is'
                                                                                'args.n_cpus * rollout_scale_factor')
    
    # arguments for flow
    parser.add_argument('--render', action='store_true', help='Show sumo-gui of results')
    parser.add_argument('--horizon', type=int, default=2000, help='Horizon of the environment')
    parser.add_argument('--sim_step', type=float, default=0.5, help='dt of a timestep')
    parser.add_argument('--low_inflow', type=int, default=800, help='the lowest inflow to sample from')
    parser.add_argument('--high_inflow', type=int, default=200, help='the highest inflow to sample from')
    parser.add_argument('--av_frac', type=float, default=0.1, help='What fraction of the vehicles should be autonomous')
    parser.add_argument('--scaling', type=int, default=1, help='How many lane should we start with. Value of 1 -> 4, '
                                                               '2 -> 8, etc.')
    parser.add_argument('--lc_on', action='store_true', help='If true, lane changing is enabled.')
    parser.add_argument('--congest_penalty', action='store_true', help='If true, an additional penalty is added '
                                                                       'for vehicles queueing in the bottleneck')
    parser.add_argument('--communicate', action='store_true', help='If true, the agents have an additional action '
                                                                   'which consists of sending a discrete signal '
                                                                   'to all nearby vehicles')
    parser.add_argument('--central_obs', action='store_true', help='If true, all agents receive the same '
                                                                   'aggregate statistics')
    parser.add_argument('--aggregate_info', action='store_true', help='If true, agents receive some '
                                                                      'centralized info')
    parser.add_argument('--congest_penalty_start', type=int, default=30, help='If congest_penalty is true, this '
                                                                              'sets the number of vehicles in edge 4'
                                                                              'at which the penalty sets in')
    
    # arguments for ray
    parser.add_argument('--use_lstm', action='store_true')
    return parser