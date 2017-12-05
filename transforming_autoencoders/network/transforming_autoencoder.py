import tensorflow as tf
from transforming_autoencoders.network.capsule import Capsule


class TransformingAutoencoder:

    def __init__(self, x, target, extra_input, input_dim, recognizer_dim, generator_dim, num_capsules):

        # Placeholders
        self.x           = x
        self.target      = target
        self.extra_input = extra_input

        # Hyper-parameters
        self.num_capsules   = num_capsules
        self.input_dim      = input_dim
        self.recognizer_dim = recognizer_dim
        self.generator_dim  = generator_dim

        self.capsules = []

        self._inference = None
        self._loss      = None
        self._summaries = []

        self.inference
        self.loss
        self.summaries

    @property
    def inference(self):
        if self._inference is None:

            # Initialize the list of capsules, each uniquely identified by its name
            for i in range(self.num_capsules):
                with tf.variable_scope('capsule_{}'.format(i)):
                    self.capsules.append(Capsule(name='capsule_{:03d}'.format(i),
                                                 x=self.x, extra_input=self.extra_input, input_dim=self.input_dim,
                                                 recognizer_dim=self.recognizer_dim, generator_dim=self.generator_dim))
            capsules_outputs = [capsule.inference for capsule in self.capsules]

            # Sum element-wise the output from all capsules
            self._inference = tf.sigmoid(tf.add_n(capsules_outputs))

        return self._inference

    @property
    def loss(self):
        if self._loss is None:
            batch_squared_error = tf.reduce_sum(tf.square(tf.subtract(self.inference, self.target)), axis=1)
            self._loss = tf.reduce_mean(batch_squared_error)
        return self._loss

    @property
    def summaries(self):
        if not self._summaries:

            self._summaries.append(tf.summary.scalar('autoencoder_loss', self.loss))

            # Visualize autoencoder input, target and prediction
            self._summaries.append(tf.summary.image('input',     tf.reshape(self.x, [-1, 28, 28, 1])))
            self._summaries.append(tf.summary.image('target',    tf.reshape(self.target, [-1, 28, 28, 1])))
            self._summaries.append(tf.summary.image('inference', tf.reshape(self.inference, [-1, 28, 28, 1])))

            # Visualize the output of each capsule singularly
            for capsule in self.capsules:
                self._summaries.extend(capsule.summaries)

            # Visualize the output of all capsules in a grid
            capsules_outputs_reshaped = [tf.reshape(capsule.inference, [-1, 28, 28, 1]) for capsule in self.capsules]
            concatenated_output = tf.concat(capsules_outputs_reshaped, axis=2)
            self._summaries.append(tf.summary.image('concatenated_capsules', concatenated_output))

        return self._summaries
