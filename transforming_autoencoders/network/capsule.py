import tensorflow as tf


dense   = tf.layers.dense
sigmoid = tf.nn.sigmoid


class Capsule(object):

    def __init__(self, name, x, extra_input, input_dim, recognizer_dim, generator_dim):

        self.name = name

        # Hyper-parameters
        self.input_dim      = input_dim
        self.recognizer_dim = recognizer_dim
        self.generator_dim  = generator_dim

        # Placeholders
        self.x           = x
        self.extra_input = extra_input

        self._inference = None
        self._summaries = []

        self.inference

    @property
    def inference(self):
        if self._inference is None:

            recognition = dense(self.x, units=self.recognizer_dim, activation=sigmoid, name='recognition_layer')

            xy_vec = dense(recognition, units=2, activation=None, name='xy_prediction')

            probability = dense(recognition, units=1, activation=sigmoid, name='probability')
            probability = tf.tile(probability, [1, self.input_dim])  # replicate probability s.t. it has input shape

            xy_extend = tf.add(xy_vec, self.extra_input)
            generation = dense(xy_extend, units=self.generator_dim, activation=sigmoid, name='generator_layer')

            out = dense(generation, units=self.input_dim, activation=None, name='output')

            self._inference = tf.multiply(out, probability)

        return self._inference

    @property
    def summaries(self):
        if not self._summaries:
            output_reshaped = tf.reshape(self.inference, [-1, 28, 28, 1])
            self._summaries.append(tf.summary.image('{}_output'.format(self.name), output_reshaped))

        return self._summaries
