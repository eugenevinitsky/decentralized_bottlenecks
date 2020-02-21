"""Multi-agent Bottleneck example.
In this example, each agent is given a single acceleration per timestep.

The agents all share a single model.
"""
from copy import deepcopy
from datetime import datetime
import json
import sys

import numpy as np
import pytz
import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune import run
from ray.tune.registry import register_env

from flow.agents.centralized_PPO import CentralizedCriticModel, CentralizedCriticModelRNN
from flow.agents.centralized_PPO import CCTrainer
from flow.agents.centralized_imitation_PPO import CCImitationTrainer

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import TrafficLightParams
from flow.core.params import VehicleParams
from flow.controllers import RLController, ContinuousRouter, \
    SimLaneChangeController
from flow.models.GRU import GRU
from flow.utils.parsers import get_multiagent_bottleneck_parser
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder


# TODO(@evinitsky) clean this up
EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def setup_rllib_params(args):
    # time horizon of a single rollout
    horizon = args.horizon
    # number of parallel workers
    n_cpus = args.n_cpus
    # number of rollouts per training iteration scaled by how many sets of rollouts per iter we want
    n_rollouts = int(args.n_cpus * args.rollout_scale_factor)
    return {'horizon': horizon, 'n_cpus': n_cpus, 'n_rollouts': n_rollouts}


def setup_flow_params(args):
    DISABLE_TB = True
    DISABLE_RAMP_METER = True
    av_frac = args.av_frac
    if args.lc_on:
        lc_mode = 1621
    else:
        lc_mode = 0

    vehicles = VehicleParams()
    if not np.isclose(av_frac, 1):
        vehicles.add(
            veh_id="human",
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=31,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=lc_mode,
            ),
            num_vehicles=1)
        vehicles.add(
            veh_id="av",
            acceleration_controller=(RLController, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=31,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=1)
    else:
        vehicles.add(
            veh_id="av",
            acceleration_controller=(RLController, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode=31,
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=1)

    # flow rate
    flow_rate = 1900 * args.scaling

    controlled_segments = [('1', 1, False), ('2', 2, True), ('3', 2, True),
                           ('4', 2, True), ('5', 1, False)]
    num_observed_segments = [('1', 1), ('2', 3), ('3', 3), ('4', 3), ('5', 1)]
    additional_env_params = {
        'target_velocity': 40,
        'disable_tb': True,
        'disable_ramp_metering': True,
        'controlled_segments': controlled_segments,
        'symmetric': False,
        'observed_segments': num_observed_segments,
        'reset_inflow': True,
        'lane_change_duration': 5,
        'max_accel': 3,
        'max_decel': 3,
        'inflow_range': [args.low_inflow, args.high_inflow],
        'start_inflow': flow_rate,
        'congest_penalty': args.congest_penalty,
        'communicate': args.communicate,
        "centralized_obs": args.central_obs,
        "aggregate_info": args.aggregate_info,
        "av_frac": args.av_frac,
        "congest_penalty_start": args.congest_penalty_start,
        "lc_mode": lc_mode,
        "life_penalty": args.life_penalty,
        'keep_past_actions': args.keep_past_actions,
        "num_sample_seconds": args.num_sample_seconds,
        "speed_reward": args.speed_reward,
        'fair_reward': False,  # This doesn't do anything, remove
        'exit_history_seconds': 0,  # This doesn't do anything, remove

        # parameters for the staggering controller that we imitate
        "n_crit": 8,
        "q_max": 15000,
        "q_min": 200,
        "q_init": 600, #
        "feedback_coeff": 1, #
        'num_imitation_iters': args.num_imitation_iters,
    }

    # percentage of flow coming out of each lane
    inflow = InFlows()
    if not np.isclose(args.av_frac, 1.0):
        inflow.add(
            veh_type='human',
            edge='1',
            vehs_per_hour=flow_rate * (1 - args.av_frac),
            departLane='random',
            departSpeed=10.0)
        inflow.add(
            veh_type='av',
            edge='1',
            vehs_per_hour=flow_rate * args.av_frac,
            departLane='random',
            departSpeed=10.0)
    else:
        inflow.add(
            veh_type='av',
            edge='1',
            vehs_per_hour=flow_rate,
            departLane='random',
            departSpeed=10.0)

    traffic_lights = TrafficLightParams()
    if not DISABLE_TB:
        traffic_lights.add(node_id='2')
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id='3')

    additional_net_params = {'scaling': args.scaling, "speed_limit": 23.0}

    if args.imitate:
        env_name = 'MultiBottleneckImitationEnv'
    else:
        env_name = 'MultiBottleneckEnv'
    flow_params = dict(
        # name of the experiment
        exp_tag=args.exp_title,

        # name of the flow environment the experiment is running on
        env_name=env_name,

        # name of the scenario class the experiment is running on
        scenario='BottleneckScenario',

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=args.sim_step,
            render=args.render,
            print_warnings=False,
            restart_instance=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            warmup_steps=int(50 / args.sim_step),
            sims_per_step=2,
            horizon=args.horizon,
            clip_actions=False,
            additional_params=additional_env_params,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # scenario's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.vehicles.Vehicles)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing='uniform',
            min_gap=5,
            lanes_distribution=float('inf'),
            edges_distribution=['2', '3', '4', '5'],
        ),

        # traffic lights to be introduced to specific nodes (see
        # flow.core.traffic_lights.TrafficLights)
        tls=traffic_lights,
    )
    return flow_params


def on_episode_end(info):
    env = info['env'].get_unwrapped()[0]
    outflow_over_last_500 = env.k.vehicle.get_outflow_rate(int(500 / env.sim_step))
    inflow = env.inflow
    # round it to 100
    inflow = int(inflow / 100) * 100
    episode = info["episode"]
    episode.custom_metrics["net_outflow_{}".format(inflow)] = outflow_over_last_500

def on_train_result(info):
    """Store the mean score of the episode, and increment or decrement how many adversaries are on"""
    result = info["result"]
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.set_iteration_num(result['training_iteration'])))

def setup_exps(args):
    rllib_params = setup_rllib_params(args)
    flow_params = setup_flow_params(args)
    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = rllib_params['n_cpus']
    config['train_batch_size'] = args.horizon * rllib_params['n_rollouts']
    config['sgd_minibatch_size'] = min(2000, config['train_batch_size'])
    if args.use_lstm:
        config['vf_loss_coeff'] = args.vf_loss_coeff
    config['gamma'] = 0.995  # discount rate
    config['horizon'] = args.horizon
    config["batch_mode"] = "truncate_episodes"
    config["sample_batch_size"] = args.horizon
    config["observation_filter"] = "MeanStdFilter"

    # Grid search things
    if args.grid_search:
        config['num_sgd_iter'] = tune.grid_search([30, 100])
        config['lr'] = tune.grid_search([5e-6, 5e-5, 5e-4])

    # LSTM Things
    if args.use_lstm and args.use_gru:
        sys.exit("You should not specify both an LSTM and a GRU")
    if args.use_lstm:
        if args.grid_search:
            config['model']["max_seq_len"] = tune.grid_search([10, 20])
        else:
            config['model']["max_seq_len"] = 20
        config['model'].update({'fcnet_hiddens': []})
        config['model']["lstm_cell_size"] = 64
        config['model']['lstm_use_prev_action_reward'] = True
    elif args.use_gru:
        if args.grid_search:
            config['model']["max_seq_len"] = tune.grid_search([20, 40])
        else:
            config['model']["max_seq_len"] = 20
            config['model'].update({'fcnet_hiddens': []})
        model_name = "GRU"
        ModelCatalog.register_custom_model(model_name, GRU)
        config['model']['custom_model'] = model_name
        config['model']['custom_options'].update({"cell_size": 64, 'use_prev_action': True})
    else:
        config['model'].update({'fcnet_hiddens': [64, 64]})
        # model_name = "FeedForward"
        # ModelCatalog.register_custom_model(model_name, FeedForward)
        # config['model']['custom_model'] = model_name
        # config['model']['custom_options'].update({'use_prev_action': True})

    # model setup for the centralized case
    # Set up model
    if args.centralized_vf:
        if args.use_lstm:
            ModelCatalog.register_custom_model("cc_model", CentralizedCriticModelRNN)
        else:
            ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)
        config['model']['custom_model'] = "cc_model"
        config['model']['custom_options']['central_vf_size'] = args.central_vf_size
        config['model']['custom_options']['max_num_agents'] = args.max_num_agents

    if args.imitate:
        config['model']['custom_options'].update({"imitation_weight": 1e0})
        config['model']['custom_options'].update({"num_imitation_iters": args.num_imitation_iters})
        config['model']['custom_options']['hard_negative_mining'] = args.hard_negative_mining
        config["model"]["custom_options"]["final_imitation_weight"] = args.final_imitation_weight

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    test_env = create_env()
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    # Setup PG with an ensemble of `num_policies` different policy graphs
    policy_graphs = {'av': (None, obs_space, act_space, {})}

    def policy_mapping_fn(_):
        return 'av'

    config.update({
        'multiagent': {
            'policies': policy_graphs,
            'policy_mapping_fn': tune.function(policy_mapping_fn),
            "policies_to_train": ["av"]
        }
    })
    return alg_run, env_name, config


if __name__ == '__main__':
    parser = get_multiagent_bottleneck_parser()
    args = parser.parse_args()

    alg_run, env_name, config = setup_exps(args)
    if args.multi_node:
        ray.init(redis_address='localhost:6379')
    elif args.local_mode:
        ray.init(local_mode=True)
    else:
        ray.init()
    eastern = pytz.timezone('US/Eastern')
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(pytz.timezone('US/Pacific')).strftime("%m-%d-%Y")
    s3_string = "s3://eugene.experiments/trb_bottleneck_paper/" \
                + date + '/' + args.exp_title
    config['env'] = env_name

    # store custom metrics
    if args.imitate:
        config["callbacks"] = {"on_episode_end": tune.function(on_episode_end),
                               "on_train_result": tune.function(on_train_result)}

    else:
        config["callbacks"] = {"on_episode_end": tune.function(on_episode_end)}

    if args.imitate and not args.centralized_vf:
        from flow.agents.ImitationPPO import ImitationTrainer
        alg_run = ImitationTrainer
    elif args.imitate and args.centralized_vf:
        alg_run = CCImitationTrainer
    elif not args.imitate and args.centralized_vf:
        alg_run = CCTrainer

    exp_dict = {
            'name': args.exp_title,
            'run_or_experiment': alg_run,
            'checkpoint_freq': args.checkpoint_freq,
            'stop': {
                'training_iteration': args.num_iters
            },
            'config': config,
            'num_samples': args.num_samples,
        }
    if args.use_s3:
        exp_dict['upload_dir'] = s3_string

    run(**exp_dict, queue_trials=False)
