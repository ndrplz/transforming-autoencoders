import numpy as np
import tensorflow as tf


dense   = tf.layers.dense
sigmoid = tf.nn.sigmoid


class Capsule(object):

    def __init__(self, name, x, extra_input, input_shape, recognizer_dim, generator_dim, transformation):

        self.name = name

        # Hyper-parameters
        self.input_shape    = input_shape
        self.input_dim      = np.prod(self.input_shape)
        self.recognizer_dim = recognizer_dim
        self.generator_dim  = generator_dim

        # Transformation applied (whether 'translation' or 'affine')
        self.transformation = transformation

        # Placeholders
        self.x           = x
        self.extra_input = extra_input

        self._inference = None
        self._summaries = []

        self.inference

    @property
    def inference(self):
        if self._inference is None:

            x_flat = tf.layers.flatten(self.x)

            recognition = dense(x_flat, units=self.recognizer_dim, activation=sigmoid, name='recognition_layer')

            probability = dense(recognition, units=1, activation=sigmoid, name='probability')
            probability = tf.tile(probability, [1, self.input_dim])  # replicate probability s.t. it has input dim

            if self.transformation == 'translation':
                learnt_transformation = dense(recognition, units=2, activation=None, name='xy_prediction')
                learnt_transformation_extended = tf.add(learnt_transformation, self.extra_input)
            else:  # self.transformation == 'affine'
                learnt_transformation = dense(recognition, units=9, activation=None, name='xy_prediction')
                learnt_transformation = tf.reshape(learnt_transformation, shape=[-1, 3, 3])
                learnt_transformation_extended = tf.matmul(learnt_transformation, self.extra_input)
                learnt_transformation_extended = tf.layers.flatten(learnt_transformation_extended)
            generation = dense(learnt_transformation_extended,
                               units=self.generator_dim, activation=sigmoid, name='generator_layer')

            out_flat = dense(generation, units=self.input_dim, activation=None, name='output')
            out_flat = tf.multiply(out_flat, probability)

            self._inference = tf.reshape(out_flat, shape=[-1] + list(self.input_shape))

        return self._inference

    @property
    def summaries(self):
        if not self._summaries:
            self._summaries.append(tf.summary.image('{}_output'.format(self.name), self.inference))

        return self._summaries
