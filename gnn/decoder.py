import numpy as np
import tensorflow as tf

from .modules import *
from .utils import fc_matrix


def mlp_decoder_one_step(prev_step, edge_type, edge_sources, edge_targets, params, scope=None, training=False):
    """Predict the next step."""
    # prev_step shape [num_sims, time_steps, num_agents, ndims]
    ndims = prev_step.shape.as_list()[-1]

    num_types = edge_type.shape.as_list()[-2]

    with tf.variable_scope(scope) as inner_scope:
        with tf.name_scope(inner_scope.original_name_scope):
            # Send encoded state to edges.
            with tf.name_scope('node_to_edge'):
                msg_from_source = tf.transpose(tf.tensordot(prev_step, edge_sources,
                                                            axes=[[2], [1]]),
                                               perm=[0, 1, 3, 2])
                msg_from_target = tf.transpose(tf.tensordot(prev_step, edge_targets,
                                                            axes=[[2], [1]]),
                                               perm=[0, 1, 3, 2])
                # msg_from_source and msg_from_target in shape [num_sims, time_steps, num_edges, ndims]
                msg_edge = tf.concat([msg_from_source, msg_from_target], axis=-1)

            encoded_msg_by_type = []
            # Encode edge message by types and concatenate them.
            start = 1 if params['skip_zero'] else 0
            for i in range(start, num_types):
                # mlp_encoder for one edge type.
                encoded_msg = mlp_layers(msg_edge,
                                         params['hidden_units'],
                                         params['dropout'],
                                         params['batch_norm'],
                                         training=training,
                                         name='edge_MLP_encoder_{}'.format(i))

                encoded_msg_by_type.append(encoded_msg)

            encoded_msg_by_type = tf.stack(encoded_msg_by_type, axis=3)
            # shape [num_sims, time_steps, num_edges, num_types, out_units]

            with tf.name_scope('edge_encoding_avg'):
                # Sum of the edge encoding from all possible types.
                encoded_msg_sum = tf.reduce_sum(tf.multiply(encoded_msg_by_type,
                                                            edge_type[:, :, :, start:, :]),
                                                axis=3)
                # shape [num_sims, time_steps, num_edges, out_units]

            # Aggregate msg from all edges to target node.
            with tf.name_scope('edge_to_node'):
                msg_aggregated = tf.transpose(tf.tensordot(encoded_msg_sum, edge_targets,
                                                           axes=[[2], [0]]),
                                              perm=[0, 1, 3, 2])
                # shape [num_sims, time_steps, num_agents, out_units]

                # Skip connection.
                msg_node = tf.concat([prev_step, msg_aggregated], axis=-1)
                # shape [num_sims, time_steps, num_edges, 2*ndims + out_units]

            # MLP encoder
            msg_node_encoded = mlp_layers(msg_node,
                                          params['hidden_units'],
                                          params['dropout'],
                                          batch_norm=False,
                                          training=training,
                                          name='node_state_MLP_decoder')

            pred_state = tf.layers.dense(msg_node_encoded, ndims, name='linear')

            return pred_state


def mlp_decoder_multisteps(features, params, pred_steps, training=False):
    time_series = features['time_series']
    # time_series shape [num_sims, time_steps, num_agents, ndims]
    # edge_type shape [num_sims, num_edges, num_edge_types]
    num_sims, time_steps, num_agents, ndims = time_series.shape.as_list()

    with tf.name_scope('edge_connection'):
        edge_sources, edge_targets = np.where(fc_matrix(num_agents))
        # One-hot representation of indices of edge sources and targets.
        # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
        edge_sources = tf.one_hot(edge_sources, num_agents)
        edge_targets = tf.one_hot(edge_targets, num_agents)

    with tf.name_scope('edge_type'):
        edge_type = features['edge_type']
        # Expand edge_type so that it has same number of dimensions as time_series.
        edge_type = tf.expand_dims(edge_type, 1)
        edge_type = tf.expand_dims(edge_type, 4)
        # edge_type shape [num_sims, 1, num_edges, num_edge_types, 1]

    with tf.variable_scope('decoder_one_step') as scope:
        starting_points = tf.expand_dims(time_series, 2)
        # Shape [num_sims, time_steps, 1, num_agents, ndims]

    def decoder_one_step(pred_time_series):
        prev_step = pred_time_series[:, :, -1, :, :]
        with tf.name_scope(scope.original_name_scope):
            next_step = prev_step + \
                mlp_decoder_one_step(prev_step, edge_type, edge_sources,
                                     edge_targets, params, scope, training)

            pred_time_series = tf.concat([pred_time_series, tf.expand_dims(next_step, 2)],
                                         axis=2)

        return pred_time_series

    i = 0
    pred_time_series = starting_points
    _, pred_time_series = tf.while_loop(
        lambda i, _: i < pred_steps,
        lambda i, pred_time_series: (i+1, decoder_one_step(pred_time_series)),
        [i, pred_time_series],
        shape_invariants=[tf.TensorShape(None),
                          tf.TensorShape([num_sims, time_steps, None, num_agents, ndims])])
    # pred_time_series Shape [num_sims, num_starting_points, pred_steps, num_agents, ndims]

    # Ignore the 0th pred step, which is the starting point.
    return pred_time_series[:, :, 1:, :, :]


decoder_fn = {
    'mlp': mlp_decoder_multisteps
}
