import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def load_train_data():
    mnist = input_data.read_data_sets('data', one_hot=True)
    return mnist.train.images


def load_validation_data():
    mnist = input_data.read_data_sets('data', one_hot=True)
    return mnist.validation.images


def translate_randomly(x, max_offset):
    X = np.reshape(x, (len(x), 28, 28))
    X_trans, trans, X_original = [], [], []

    for i in np.random.permutation(len(X)):
        trans_x = random.randint(-max_offset, max_offset)
        trans_y = random.randint(-max_offset, max_offset)

        trans_img = np.roll(np.roll(X[i], trans_x, axis=0), trans_y, axis=1)
        X_trans.append(trans_img.flatten())
        X_original.append(x[i])
        trans.append((trans_x, trans_y))

    return np.array(X_trans), np.array(trans), np.array(X_original)
