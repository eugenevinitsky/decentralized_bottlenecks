"""Example of using a custom RNN keras model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.recurrent_tf_modelv2 import RecurrentTFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class GRU(RecurrentTFModelV2):
    """Simple custom gated recurrent unit."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,):
        super(GRU, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)

        # Define input layers
        input_layer = tf.keras.layers.Input(
            shape=(None, obs_space.shape[0]), name="inputs")
        # Preprocess observations with the appropriate number of hidden layers
        last_layer = input_layer
        i = 1
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens")
        for size in hiddens:
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            i += 1

        cell_size = model_config["custom_options"].get("cell_size")
        self.cell_size = cell_size
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size, ), name="h")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in")

        gru_out, state_h = tf.keras.layers.GRU(
            self.cell_size, return_sequences=True, return_state=True, name="gru")(
                inputs=last_layer,
                mask=tf.sequence_mask(seq_in),
                initial_state=[state_in_h])

        # Postprocess LSTM output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs,
            activation=tf.keras.activations.linear,
            name="logits")(gru_out)
        values = tf.keras.layers.Dense(
            1, activation=None, name="values")(gru_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h],
            outputs=[logits, values, state_h])
        self.register_variables(self.rnn_model.variables)
        self.rnn_model.summary()

    @override(RecurrentTFModelV2)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h = self.rnn_model([inputs, seq_lens] +
                                                          state)
        return model_out, [h]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])