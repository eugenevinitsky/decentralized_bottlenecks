from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, SlimFC, \
    _get_activation_fn
from ray.rllib.utils.annotations import override

logger = logging.getLogger(__name__)


class FeedForward(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        hiddens = model_config.get("fcnet_hiddens")
        logger.debug("Constructing fcnet {}".format(hiddens))
        layers = []
        last_layer_size = np.product(obs_space.shape)
        for size in hiddens:
            layers.append(nn.Linear(in_features=last_layer_size, out_features=size))
            layers.append(nn.ReLU())
            last_layer_size = size

        self._hidden_layers = nn.Sequential(*layers)

        self._hidden_layers.apply(init_weights)

        # TODO(@ev) pick the right initialization
        self._logits = nn.Linear(
            in_features=last_layer_size,
            out_features=num_outputs)

        self._logits.apply(large_initializer)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        features = self._hidden_layers(obs.reshape(obs.shape[0], -1))
        logits = self._logits(features)
        return logits, state


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def large_initializer(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(-500.0)

