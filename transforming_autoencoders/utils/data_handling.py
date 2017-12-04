import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def load_MNIST_data():
    mnist = input_data.read_data_sets('data', one_hot=True)
    return {'train': mnist.train.images,
            'validation': mnist.validation.images,
            'test': mnist.test.images}


def translate_randomly(x, max_offset):
    x = np.reshape(x, (len(x), 28, 28))
    x_translated, translations, x_original = [], [], []

    for i in np.random.permutation(len(x)):
        trans_x = random.randint(-max_offset, max_offset)
        trans_y = random.randint(-max_offset, max_offset)

        trans_img = np.roll(np.roll(x[i], trans_x, axis=0), trans_y, axis=1)
        x_translated.append(trans_img.flatten())
        x_original.append(x[i].flatten())
        translations.append((trans_x, trans_y))

    return {'x_original': np.array(x_original),
            'x_translated': np.array(x_translated),
            'translations': np.array(translations)}
