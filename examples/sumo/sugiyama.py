"""Used as an example of sugiyama experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""

from flow.controllers import IDMController, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, \
    InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.networks.ring import RingNetwork, ADDITIONAL_NET_PARAMS


def sugiyama_example(render=None):
    """
    Perform a simulation of vehicles on a ring road.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=22)

    flow_params = dict(
        # name of the experiment
        exp_tag='ring',

        # name of the flow environment the experiment is running on
        env_name=AccelEnv,

        # name of the network class the experiment is running on
        network=RingNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            render=True,
            sim_step=0.1,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=1500,
            additional_params=ADDITIONAL_ENV_PARAMS,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params=ADDITIONAL_NET_PARAMS.copy(),
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            bunching=20,
        ),
    )

    return Experiment(flow_params, None)


if __name__ == "__main__":
    # import the experiment variable
    exp = sugiyama_example()

    # run for a set number of rollouts / time steps
    exp.run(1)
