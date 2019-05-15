import numpy as np
import matplotlib.pyplot as plt
import time

import argparse
import os

def main():

    pose = np.load(os.path.join(ARGS.data_dir, 'prediction_1.npy')) #"chaser_20_not_random/test_random_position.npy")

    if ARGS.show_simulation:

        for i in range(pose.shape[0]):

            plt.clf()
            plt.axis([-1.6, 1.6, -1, 1])

            plt.scatter(np.squeeze(pose[i,:,:,0]),np.squeeze(pose[i,:,:,1]))

            plt.pause(0.3)

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--show-simulation', action = 'store_true', default = True,
                        help='data directory')

    ARGS = parser.parse_args()


    main()
