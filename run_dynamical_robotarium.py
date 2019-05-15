import os
import argparse
import json
import time


import tensorflow as tf
import numpy as np

import gnn
from gnn.data import load_data
from gnn.utils import gumbel_softmax


import rps.robotarium as robotarium
import rps.utilities.graph as graph
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *


def model_fn(features, labels, mode, params):
    pred_stack = gnn.dynamical.dynamical_multisteps(features,
                                                    params,
                                                    params['pred_steps'],
                                                    training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'next_steps': pred_stack}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels, pred_stack)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.train.get_global_step(),
            decay_steps=100,
            decay_rate=0.99,
            staircase=True,
            name='learning_rate'
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Use the loss between adjacent steps in original time_series as baseline
    time_series_loss_baseline = tf.metrics.mean_squared_error(features['time_series'][:, 1:, :, :],
                                                              features['time_series'][:, :-1, :, :])

    eval_metric_ops = {'time_series_loss_baseline': time_series_loss_baseline}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(features, seg_len, pred_steps, batch_size, mode='train'):

    time_series = features['time_series']
    num_sims, time_steps, num_agents, ndims = time_series.shape
    # Shape [num_sims, time_steps, num_agents, ndims]
    time_series_stack = gnn.utils.stack_time_series(time_series[:, :-pred_steps, :, :],
                                                    seg_len)
    # Shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, num_agents, ndims]
    expected_time_series_stack = gnn.utils.stack_time_series(time_series[:, seg_len:, :, :],
                                                             pred_steps)
    # Shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, num_agents, ndims]
    assert time_series_stack.shape[:2] == expected_time_series_stack.shape[:2]

    time_segs = time_series_stack.reshape([-1, seg_len, num_agents, ndims])
    expected_time_segs = expected_time_series_stack.reshape([-1, pred_steps, num_agents, ndims])

    processed_features = {'time_series': time_segs}
    if 'edge_type' in features:
        processed_features['edge_type'] = features['edge_type']
    labels = expected_time_segs

    if mode == 'train':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            y=labels,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True
        )
    elif mode == 'eval':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            y=labels,
            batch_size=batch_size,
            shuffle=False
        )
    elif mode == 'test':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            batch_size=batch_size,
            shuffle=False
        )

def apply_control(r,N, goal_velocities):

    x = r.get_poses()
    for _ in range(9):
        # just use the velocities from the neural network or the file
        x = r.get_poses()
        r.set_velocities(np.arange(N), single_integrator_to_unicycle2(goal_velocities, x))
        r.step()
        # total runtime of the for loop is matched to 0.3 sec approximately
        time.sleep(0.033)


def move_to_goal(r, N, si_barrier_cert, goal_points):

    x = r.get_poses()

    # While the number of robots at the required poses is less
    # than N...
    while (np.size(at_pose(x, goal_points, rotation_error=100)) != N):

        # Get poses of agents
        x = r.get_poses()
        x_si = x[:2, :]

        # Create single-integrator control inputs
        dxi = single_integrator_position_controller(x_si, goal_points[:2, :], magnitude_limit=0.08)

        # Create safe control inputs (i.e., no collisions)
        dxi = si_barrier_cert(dxi, x_si)

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        #r.set_velocities(np.arange(N), single_integrator_to_unicycle2(dxi, x))
        r.set_velocities(np.arange(N), single_integrator_to_unicycle2(dxi, x))
        # Iterate the simulation
        r.step()



def get_orientation(velocities):
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    cos_sin = velocities / speeds
    orientations = np.arccos(cos_sin[:, :1, :]) * np.sign(cos_sin[:, 1:, :])
    return np.nan_to_num(orientations)

def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    model_params['pred_steps'] = ARGS.pred_steps
    seg_len = 2 * len(model_params['cnn']['filters']) + 1

    cnn_multistep_regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    # Prediction
    if ARGS.test:
        if model_params.get('edge_types', 0) > 1:
            test_data, test_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                             prefix='test')
            test_edge = gnn.utils.one_hot(test_edge, model_params['edge_types'], np.float32)

            features = {'time_series': test_data, 'edge_type': test_edge}
        else:
            test_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                                  prefix='test')
            features = {'time_series': test_data}

        if not ARGS.dynamic_update:

            curr_time = time.time()

            predict_input_fn = input_fn(features, seg_len, ARGS.pred_steps, ARGS.batch_size, 'test')
            prediction = cnn_multistep_regressor.predict(input_fn=predict_input_fn)
            prediction = np.array([pred['next_steps'] for pred in prediction])

            prediction = np.swapaxes(prediction, 0, 1)

            print(f'GNN execution time = {- curr_time + time.time()}')


        #print(prediction.shape)


        print("===========================================================")
        print("===========================================================")

        # Instantiate Robotarium object
        N = test_data.shape[2]
        r = robotarium.Robotarium(number_of_agents=N, show_figure=True, save_data=False, update_time=0.3)

        # Create barrier certificates to avoid collision
        si_barrier_cert = create_single_integrator_barrier_certificate(N)

        # ------------------------- initalizing initial positions ------------------------

        # initialize the the agents according to the simulation
        initial_goal_points = np.squeeze(np.swapaxes(test_data,2,3)[:,0,:2,:])
        initial_orientations = np.zeros((1,20))

        initial_goal_states = np.concatenate([initial_goal_points, initial_orientations], axis=0)

        move_to_goal(r, N, si_barrier_cert, initial_goal_states)
        r.call_at_scripts_end()

        time.sleep(0.03)

        print(f'Step {1}')
        print(f'error = {np.linalg.norm(r.get_poses()[:2,:] - initial_goal_points)}')

        current_position = r.get_poses()[:2,:]
        current_velocities = unicycle_to_single_integrator(r.velocities, r.get_poses())

        current_pos_vel = np.concatenate([current_position, current_velocities], axis=0)


        pos_vel_log = np.swapaxes(np.expand_dims(np.expand_dims(current_pos_vel, axis = 0), axis = 0),2,3)



        for i in range(1,8):
            print(f'Step {i+1}')
            goal_points = np.squeeze(np.swapaxes(test_data,2,3)[:,i,:2,:])
            goal_velocities = np.squeeze(np.swapaxes(test_data,2,3)[:,i,2:,:])

            apply_control(r, N, goal_velocities)

            current_position = r.get_poses()[:2,:]
            current_velocities = unicycle_to_single_integrator(r.velocities, r.get_poses())

            current_pos_vel = np.concatenate([current_position, current_velocities], axis=0)

            pos_vel_log = np.concatenate([pos_vel_log,
                                np.swapaxes(np.expand_dims(np.expand_dims(current_pos_vel, axis = 0), axis = 0),2,3)], axis = 1)

            print(f'error = {np.linalg.norm(r.get_poses()[:2,:] - goal_points)}')

            # Always call this function at the end of your scripts!  It will accelerate the
            # execution of your experiment
            r.call_at_scripts_end()


        #------------- intialization end ----------------------

        # -------------------- Prediction from Graph neural network -------------------------

        pos_vel_log = pos_vel_log.astype(dtype=np.float32)

        for i in range(8,test_data.shape[1]):

            if ARGS.dynamic_update:

                curr_time = time.time()

                features = {'time_series': pos_vel_log[:,i-8:i,:,:]}

                predict_input_fn = input_fn(features, seg_len, ARGS.pred_steps, ARGS.batch_size, 'test')
                prediction = cnn_multistep_regressor.predict(input_fn=predict_input_fn)


                prediction = np.array([pred['next_steps'] for pred in prediction])

                print("------------------------------------------------------------------")

                print(f'GNN execution time = {-curr_time +time.time()}')

                goal_velocities = np.squeeze(np.swapaxes(prediction,2,3)[:,:,2:,:])

            else:
                print(prediction.shape)
                goal_velocities = np.squeeze(np.swapaxes(prediction,2,3)[:,i-8,2:,:])

            print(f'Step {i+1}')
            goal_points = np.squeeze(np.swapaxes(test_data,2,3)[:,i,:2,:])
            #goal_velocities = np.squeeze(np.swapaxes(prediction,2,3)[:,:,2:,:])

            apply_control(r, N, goal_velocities)

            current_position = r.get_poses()[:2,:]
            current_velocities = unicycle_to_single_integrator(r.velocities, r.get_poses())

            current_pos_vel = np.concatenate([current_position, current_velocities], axis=0)

            pos_vel_log = np.concatenate([pos_vel_log,
                                np.swapaxes(np.expand_dims(np.expand_dims(current_pos_vel, axis = 0), axis = 0),2,3)], axis = 1)

            pos_vel_log = pos_vel_log.astype(dtype=np.float32)


            print(f'error = {np.linalg.norm(r.get_poses()[:2,:] - goal_points)}')

            # Always call this function at the end of your scripts!  It will accelerate the
            # execution of your experiment
            r.call_at_scripts_end()


        # np.save(os.path.join(ARGS.log_dir, 'prediction_robotarium_{}.npy'.format(
        #     ARGS.pred_steps)), pos_vel_log)
        #
        # time.sleep(5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-transpose', type=int, nargs=4, default=None,
                        help='axes for data transposition')
    parser.add_argument('--data-size', type=int, default=None,
                        help='optional data size cap to use for training')
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--train-steps', type=int, default=1000,
                        help='number of training steps')
    parser.add_argument('--pred-steps', type=int, default=1,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='turn on logging info')
    parser.add_argument('--test', action='store_true', default=True,
                        help='turn on test')
    parser.add_argument('--dynamic-update', action='store_true', default=False,
                        help='turn on dynamic update from GNN')
    ARGS = parser.parse_args()

    if ARGS.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    main()
