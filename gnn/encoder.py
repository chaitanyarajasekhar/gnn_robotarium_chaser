import numpy as np
import tensorflow as tf

from .modules import *
from .utils import fc_matrix


def node_to_edge(node_state, edge_sources, edge_targets):
    """Propagate node states to edges."""
    with tf.name_scope("node_to_edge"):
        msg_from_source = tf.transpose(tf.tensordot(node_state, edge_sources, axes=[[1], [1]]),
                                       perm=[0, 2, 1])
        msg_from_target = tf.transpose(tf.tensordot(node_state, edge_targets, axes=[[1], [1]]),
                                       perm=[0, 2, 1])
        ## msg_from_source and msg_from_target in shape [num_sims, num_edges, out_units]
        msg_edge = tf.concat([msg_from_source, msg_from_target], axis=-1)

    return msg_edge


def edge_to_node(edge_msg, edge_targets):
    """Send edge messages to target nodes."""
    with tf.name_scope("edge_to_node"):
        node_state = tf.transpose(tf.tensordot(edge_msg, edge_targets, axes=[[1], [0]]),
                                  perm=[0, 2, 1])  # Shape [num_sims, num_agents, out_units].

    return node_state


def mlp_encoder(features, classes, params, training=False):
    # Tensor `features` has shape [num_sims, time_steps, num_agents, ndims].
    time_steps, num_agents, ndims = features.shape.as_list()[1:]
    # Input Layer
    # Transpose to [num_sims, num_agents, time_steps, ndims]
    with tf.name_scope("input_shape"):
        features = tf.transpose(features, [0, 2, 1, 3])
        state = tf.reshape(features, shape=[-1, num_agents, time_steps * ndims])
    # Node state encoder with MLP.
    node_state = mlp_layers(state,
                            params['hidden_units'],
                            params['dropout'],
                            params['batch_norm'],
                            training=training,
                            name="node_encoding_MLP_1")

    # Send encoded state to edges.
    # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
    edge_sources, edge_targets = np.where(fc_matrix(num_agents))
    # One-hot representation of indices of edge sources and targets.
    with tf.name_scope("one_hot"):
        edge_sources = tf.one_hot(edge_sources, num_agents)
        edge_targets = tf.one_hot(edge_targets, num_agents)

    # Form edges. Shape [num_sims, num_edges, 2 * hidden_units]
    msg_edge = node_to_edge(node_state, edge_sources, edge_targets)

    # Store skip.
    msg_edge_skip = msg_edge

    # Encode edge messages with MLP. Shape [num_sims, num_edges, hidden_units]
    msg_edge = mlp_layers(msg_edge,
                          params['hidden_units'],
                          params['dropout'],
                          params['batch_norm'],
                          training=training,
                          name='edge_encoding_MLP_1')

    # Compute edge influence to node. Shape [num_sims, num_agents, hidden_units]
    node_state = edge_to_node(msg_edge, edge_targets)

    # Encode node state with MLP
    node_state = mlp_layers(node_state,
                            params['hidden_units'],
                            params['dropout'],
                            params['batch_norm'],
                            training=training,
                            name='node_encoding_MLP_2')

    # Propagate node states to edges again.
    msg_edge = node_to_edge(node_state, edge_sources, edge_targets)

    # Encode edge messages with MLP.
    msg_edge = mlp_layers(msg_edge,
                          params['hidden_units'],
                          params['dropout'],
                          params['batch_norm'],
                          training=training,
                          name='edge_encoding_MLP_2')

    # Concatenate skip msg. Shape [num_sims, num_edges, 4 * hidden_units]
    msg_edge = tf.concat([msg_edge, msg_edge_skip], axis=-1)

    # Shape [num_sims, num_edges, classes]
    edge_type = tf.layers.dense(msg_edge, classes, name='edge_type_encoding')

    return edge_type


# Encoder function factory.
encoder_fn = {
    'mlp': mlp_encoder
}
