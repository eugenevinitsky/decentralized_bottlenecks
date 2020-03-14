"""File demonstrating formation of congestion in bottleneck."""

from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.networks.bottleneck import BottleneckNetwork
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.envs.bottleneck_env import BottleneckEnv
from flow.core.experiment import Experiment

import logging

import numpy as np
SCALING = 1
DISABLE_TB = True
# If set to False, ALINEA will control the ramp meter
DISABLE_RAMP_METER = True
INFLOW = 2400


class BottleneckDensityExperiment(Experiment):

    def __init__(self, flow_params, inflow=INFLOW):
        super().__init__(flow_params)
        self.inflow = inflow

    def run(self, num_runs, num_steps, end_len=500, rl_actions=None, convert_to_csv=False):
        info_dict = {}
        if rl_actions is None:

            def rl_actions(*_):
                return None

        rets = []
        mean_rets = []
        ret_lists = []
        vels = []
        mean_vels = []
        std_vels = []
        mean_densities = []
        mean_outflows = []
        lane_4_vels = []
        for i in range(num_runs):
            vel = np.zeros(num_steps)
            logging.info('Iter #' + str(i))
            ret = 0
            ret_list = []
            step_outflows = []
            step_densities = []
            state = self.env.reset()
            for j in range(num_steps):
                state, reward, done, _ = self.env.step(rl_actions(state))
                vel[j] = np.mean(self.env.k.vehicle.get_speed(
                    self.env.k.vehicle.get_ids()))
                if j >= num_steps - end_len:
                    vehicles = self.env.k.vehicle
                    vehs_on_four = vehicles.get_ids_by_edge('4')
                    lanes = vehicles.get_lane(vehs_on_four)
                    lane_dict = {veh_id: lane for veh_id, lane in
                                 zip(vehs_on_four, lanes)}
                    sort_by_lane = sorted(vehs_on_four,
                                          key=lambda x: lane_dict[x])
                    num_zeros = lanes.count(0)
                    if num_zeros > 0:
                        speed_on_zero = np.mean(vehicles.get_speed(
                            sort_by_lane[0:num_zeros]))
                    else:
                        speed_on_zero = 0.0
                    if num_zeros < len(vehs_on_four):
                        speed_on_one = np.mean(vehicles.get_speed(
                            sort_by_lane[num_zeros:]))
                    else:
                        speed_on_one = 0.0
                    lane_4_vels.append([self.inflow, speed_on_zero,
                                        speed_on_one])
                ret += reward
                ret_list.append(reward)

                env = self.env
                step_outflow = env.get_bottleneck_outflow_vehicles_per_hour(20)
                density = self.env.get_bottleneck_density()

                step_outflows.append(step_outflow)
                step_densities.append(density)
                if done:
                    break
            rets.append(ret)
            vels.append(vel)
            mean_densities.append(sum(step_densities[100:]) /
                                  (num_steps - 100))
            env = self.env
            outflow = env.get_bottleneck_outflow_vehicles_per_hour(end_len)
            mean_outflows.append(outflow)
            mean_rets.append(np.mean(ret_list))
            ret_lists.append(ret_list)
            mean_vels.append(np.mean(vel))
            std_vels.append(np.std(vel))
            print('Round {0}, return: {1}'.format(i, ret))

        info_dict['returns'] = rets
        info_dict['velocities'] = vels
        info_dict['mean_returns'] = mean_rets
        info_dict['per_step_returns'] = ret_lists
        info_dict['average_outflow'] = np.mean(mean_outflows)
        info_dict['per_rollout_outflows'] = mean_outflows
        info_dict['lane_4_vels'] = lane_4_vels

        info_dict['average_rollout_density_outflow'] = np.mean(mean_densities)

        print('Average, std return: {}, {}'.format(
            np.mean(rets), np.std(rets)))
        print('Average, std speed: {}, {}'.format(
            np.mean(mean_vels), np.std(std_vels)))
        self.env.terminate()

        return info_dict


def bottleneck_example(flow_rate, horizon, restart_instance=False,
                       render=False):
    """
    Perform a simulation of vehicles on a bottleneck.

    Parameters
    ----------
    flow_rate : float
        total inflow rate of vehicles into the bottleneck
    horizon : int
        time horizon
    restart_instance: bool, optional
        whether to restart the instance upon reset
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a bottleneck.
    """
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=25,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=1621,
        ),
        num_vehicles=1)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehsPerHour=INFLOW,
        departLane="random",
        departSpeed=10)

    traffic_lights = TrafficLightParams()
    if not DISABLE_TB:
        traffic_lights.add(node_id="2")
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id="3")

    flow_params = dict(
        # name of the experiment
        exp_tag='bay_bridge_toll',

        # name of the flow environment the experiment is running on
        env_name=BottleneckEnv,

        # name of the network class the experiment is running on
        network=BottleneckNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.5,
            render=render,
            overtake_right=False,
            restart_instance=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=horizon,
            additional_params={
                "target_velocity": 40,
                "max_accel": 1,
                "max_decel": 1,
                "lane_change_duration": 5,
                "add_rl_if_exit": False,
                "disable_tb": DISABLE_TB,
                "disable_ramp_metering": DISABLE_RAMP_METER
            }
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params={
                "scaling": SCALING,
                "speed_limit": 23
            }
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing="random",
            min_gap=5,
            lanes_distribution=float("inf"),
            edges_distribution=["2", "3", "4", "5"]
        ),

        # traffic lights to be introduced to specific nodes (see
        # flow.core.params.TrafficLightParams)
        tls=traffic_lights,
    )

    return BottleneckDensityExperiment(flow_params)


if __name__ == '__main__':
    # import the experiment variable
    # inflow, number of steps, binary
    HORIZON = 1000
    exp = bottleneck_example(INFLOW, HORIZON, render=True)
    exp.run(5, HORIZON)
