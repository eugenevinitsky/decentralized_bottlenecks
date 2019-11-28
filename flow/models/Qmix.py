"""Example of using a custom RNN keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from gym.spaces import Tuple, Discrete, Dict

import numpy as np
import argparse

import ray
from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from ray.rllib.agents.qmix.qmix_policy import QMixLoss, QMixTorchPolicy, _validate, _mac, _unroll_mac
from ray.rllib.agents.qmix.mixers import VDNMixer, QMixer
from ray.rllib.agents.qmix.model import RNNModel, _get_size
from ray.rllib.agents.qmix.qmix import make_sync_batch_optimizer, DEFAULT_CONFIG
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.policy.policy import Policy, TupleActions
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.model import _unpack_obs
from ray.rllib.env.constants import GROUP_REWARDS
from ray.rllib.utils.annotations import override

import torch as th
from torch.optim import RMSprop
from torch.distributions import Categorical
import torch.nn.functional as F


class VariableQMixLoss(QMixLoss):
    def __init__(self,
                 model,
                 target_model,
                 mixer,
                 target_mixer,
                 n_agents,
                 n_actions,
                 double_q=True,
                 gamma=0.99):
        QMixLoss.__init__(self, model, target_model, mixer, target_mixer, n_agents, n_actions, double_q, gamma)

    def forward(self, rewards, actions, terminated, mask, obs, next_obs,
                action_mask, next_action_mask, valid_agents, next_valid_agents):
        """Forward pass of the loss.

        Arguments:
            rewards: Tensor of shape [B, T, n_agents]
            actions: Tensor of shape [B, T, n_agents]
            terminated: Tensor of shape [B, T, n_agents]
            mask: Tensor of shape [B, T, n_agents]
            obs: Tensor of shape [B, T, n_agents, obs_size]
            next_obs: Tensor of shape [B, T, n_agents, obs_size]
            action_mask: Tensor of shape [B, T, n_agents, n_actions]
            next_action_mask: Tensor of shape [B, T, n_agents, n_actions]
        """

        # Calculate estimated Q-Values
        mac_out = _unroll_mac(self.model, obs)

        # Pick the Q-Values for the actions taken -> [B * n_agents, T]
        chosen_action_qvals = th.gather(
            mac_out, dim=3, index=actions.unsqueeze(3)).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = _unroll_mac(self.target_model, next_obs)

        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf

        # Max over target Q-Values
        if self.double_q:
            # Double Q learning computes the target Q values by selecting the
            # t+1 timestep action according to the "policy" neural network and
            # then estimating the Q-value of that action with the "target"
            # neural network

            # Compute the t+1 Q-values to be used in action selection
            # using next_obs
            mac_out_tp1 = _unroll_mac(self.model, next_obs)

            # mask out unallowed actions
            mac_out_tp1[ignore_action_tp1] = -np.inf

            # obtain best actions at t+1 according to policy NN
            cur_max_actions = mac_out_tp1.max(dim=3, keepdim=True)[1]

            # use the target network to estimate the Q-values of policy
            # network's selected actions
            target_max_qvals = th.gather(target_mac_out, 3,
                                         cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        assert target_max_qvals.min().item() != -np.inf, \
            "target_max_qvals contains a masked action; \
            there may be a state with no valid actions."

        # Mix
        if self.mixer is not None:
            # TODO(ekl) add support for handling global state? This is just
            # treating the stacked agent obs as the state.
            valid_agent_qvals = th.mul(chosen_action_qvals, valid_agents)
            next_valid_agent_qvals = th.mul(target_max_qvals, next_valid_agents)
            chosen_action_qvals = self.mixer(valid_agent_qvals, obs) # pass valid agents to mixer
            target_max_qvals = self.target_mixer(next_valid_agent_qvals, next_obs) # pass next valid agents to mixer

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error**2).sum() / mask.sum()
        return loss, mask, masked_td_error, chosen_action_qvals, targets

class VariableQMixTorchPolicy(QMixTorchPolicy):
    """ QMix Torch Policy for handling a variable number of 
    agents
    """

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.qmix.qmix.DEFAULT_CONFIG, **config)
        self.config = config
        self.observation_space = obs_space
        self.action_space = action_space
        self.n_agents = len(obs_space.original_space.spaces)
        self.n_actions = action_space.spaces[0].n
        self.h_size = config["model"]["lstm_cell_size"]

        agent_obs_space = obs_space.original_space.spaces[0]
        assert isinstance(agent_obs_space, Dict)

        space_keys = set(agent_obs_space.spaces.keys())
        if space_keys != {"obs", "valid_agent"}: # force obs space to be dict with only valid agent and obs
            raise ValueError(
                "Dict obs space for agent must have keyset "
                "['obs', 'valid_agent'], got {}".format(space_keys))

        assert agent_obs_space.spaces["valid_agent"] == Discrete(2)

        self.has_action_mask = False # action mask should never exist
        self.obs_size = _get_size(agent_obs_space.spaces["obs"])
        # The real agent obs space is nested inside the dict
        agent_obs_space = agent_obs_space.spaces["obs"]

        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=RNNModel)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="target_model",
            default_model=RNNModel)

        # Setup the mixer network.
        # The global state is just the stacked agent observations for now.
        self.state_shape = [self.obs_size, self.n_agents]
        if config["mixer"] is None:
            self.mixer = None
            self.target_mixer = None
        elif config["mixer"] == "qmix":
            self.mixer = VariableQMixer(self.n_agents, self.state_shape,
                                config["mixing_embed_dim"]) # use custom VariableQMixer
            self.target_mixer = VariableQMixer(self.n_agents, self.state_shape,
                                       config["mixing_embed_dim"]) # use custom VariableQMixer
        elif config["mixer"] == "vdn":
            self.mixer = VDNMixer()
            self.target_mixer = VDNMixer()
        else:
            raise ValueError("Unknown mixer type {}".format(config["mixer"]))

        self.cur_epsilon = 1.0
        self.update_target()  # initial sync

        # Setup optimizer
        self.params = list(self.model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())
        self.loss = VariableQMixLoss(self.model, self.target_model, self.mixer,
                             self.target_mixer, self.n_agents, self.n_actions,
                             self.config["double_q"], self.config["gamma"])
        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"])

    def _unpack_observation(self, obs_batch):
        """Unpacks the action mask / tuple obs from agent grouping.

        Returns:
            obs (Tensor): flattened obs tensor of shape [B, n_agents, obs_size]
            mask (Tensor): action mask, if any
        """
        unpacked = _unpack_obs(
            np.array(obs_batch),
            self.observation_space.original_space,
            tensorlib=np)
        obs = np.concatenate(
            [o["obs"] for o in unpacked],
            axis=1).reshape([len(obs_batch), self.n_agents, self.obs_size])
        # process valid agents obs: note, we're using the second column, because we're passing a boolean as a discrete (second column is val = 1)
        valid_agents = np.array([o["valid_agent"][:,1] for o in unpacked]).reshape([len(obs_batch), self.n_agents])  # process valid agents obs
        action_mask = np.ones(
                [len(obs_batch), self.n_agents, self.n_actions]) # dummy action mask, so we don't have to re-write things

        return obs, action_mask, valid_agents

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        obs_batch, action_mask, valid_agents = self._unpack_observation(obs_batch) # get valid agents

        # Compute actions
        with th.no_grad():
            q_values, hiddens = _mac(
                self.model, th.from_numpy(obs_batch),
                [th.from_numpy(np.array(s)) for s in state_batches])
            avail = th.from_numpy(action_mask).float()
            masked_q_values = q_values.clone()
            masked_q_values[avail == 0.0] = -float("inf")
            # epsilon-greedy action selector
            random_numbers = th.rand_like(q_values[:, :, 0])
            pick_random = (random_numbers < self.cur_epsilon).long()
            random_actions = Categorical(avail).sample().long()
            actions = (pick_random * random_actions +
                       (1 - pick_random) * masked_q_values.max(dim=2)[1])
            actions = actions.numpy()
            hiddens = [s.numpy() for s in hiddens]

        return TupleActions(list(actions.transpose([1, 0]))), hiddens, {}

    @override(Policy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, valid_agents = self._unpack_observation(
            samples[SampleBatch.CUR_OBS]) # get valid agents
        next_obs_batch, next_action_mask, next_valid_agents = self._unpack_observation(
            samples[SampleBatch.NEXT_OBS]) # get next valid agents
        group_rewards = self._get_group_rewards(samples[SampleBatch.INFOS])
        # These will be padded to shape [B * T, ...]
        [rew, action_mask, next_action_mask, act, dones, obs, next_obs, valid_agents, next_valid_agents], \
            initial_states, seq_lens = \
            chop_into_sequences(
                samples[SampleBatch.EPS_ID],
                samples[SampleBatch.UNROLL_ID],
                samples[SampleBatch.AGENT_INDEX], [
                    group_rewards, action_mask, next_action_mask,
                    samples[SampleBatch.ACTIONS], samples[SampleBatch.DONES],
                    obs_batch, next_obs_batch, valid_agents, next_valid_agents],
                [samples["state_in_{}".format(k)]
                 for k in range(len(self.get_initial_state()))],
                max_seq_len=self.config["model"]["max_seq_len"],
                dynamic_max=True) # also chop valid agents, next valid agents into sequences
        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr):
            new_shape = [B, T] + list(arr.shape[1:])
            return th.from_numpy(np.reshape(arr, new_shape))

        rewards = to_batches(rew).float()
        actions = to_batches(act).long()
        obs = to_batches(obs).reshape([B, T, self.n_agents,
                                       self.obs_size]).float()
        action_mask = to_batches(action_mask)
        next_obs = to_batches(next_obs).reshape(
            [B, T, self.n_agents, self.obs_size]).float()
        next_action_mask = to_batches(next_action_mask)

        valid_agents = to_batches(valid_agents)
        next_valid_agents = to_batches(next_valid_agents)

        # TODO(ekl) this treats group termination as individual termination
        terminated = to_batches(dones.astype(np.float32)).unsqueeze(2).expand(
            B, T, self.n_agents)

        # Create mask for where index is < unpadded sequence length
        filled = (np.reshape(np.tile(np.arange(T), B), [B, T]) <
                  np.expand_dims(seq_lens, 1)).astype(np.float32)
        mask = th.from_numpy(filled).unsqueeze(2).expand(B, T, self.n_agents)

        # Compute loss
        loss_out, mask, masked_td_error, chosen_action_qvals, targets = \
            self.loss(rewards, actions, terminated, mask, obs,
                      next_obs, action_mask, next_action_mask,
                      valid_agents, next_valid_agents)

        # Optimise
        self.optimiser.zero_grad()
        loss_out.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.config["grad_norm_clipping"])
        self.optimiser.step()

        mask_elems = mask.sum().item()
        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm
            if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() /
            mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }
        return {LEARNER_STATS_KEY: stats}


class VariableQMixer(QMixer):
    def __init__(self, n_agents, state_shape, mixing_embed_dim):
        super(VariableQMixer, self).__init__(n_agents, state_shape, mixing_embed_dim)


    def forward(self, agent_qs, states):
        """Forward pass for the mixer.

        Arguments:
            agent_qs: Tensor of shape [B, T, n_agents, n_actions]
            states: Tensor of shape [B, T, state_dim]
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        import ipdb; ipdb.set_trace()
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


QMixTrainer = GenericOffPolicyTrainer.with_updates(
    name="VariableQMIX",
    default_config=DEFAULT_CONFIG,
    default_policy=VariableQMixTorchPolicy,
    make_policy_optimizer=make_sync_batch_optimizer)