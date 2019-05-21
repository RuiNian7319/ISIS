"""
Importance Sampling Incremental Supervised learning (ISIS) v1.0

Rui Nian
"""

import numpy as np
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt

import gc

import warnings
warnings.filterwarnings('ignore')
gc.enable()


class ISIS:

    def __init__(self):
        pass

    def get_data_attr(self):
        pass


if __name__ == '__main__':

    rand_label = np.random.uniform(-300, 300, size=(1000, 1))
    rand_x1 = np.random.uniform(-50, 200, size=(1000, 1))
    rand_x2 = (rand_label - 2 * rand_x1) / 4

    data = np.concatenate([rand_label, rand_x1, rand_x2], axis=1)
    data[:, 0] = np.add(data[:, 0], np.random.normal(0, 50, size=(data.shape[0],)))

    del rand_label, rand_x1, rand_x2

    pred = 2 * data[:, 1] + 4 * data[:, 2]
    plt.plot(data[0:25, 0])
    plt.plot(pred[0:25])
    plt.show()
