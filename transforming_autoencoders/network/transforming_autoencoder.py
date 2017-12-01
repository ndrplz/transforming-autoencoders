import tensorflow as tf
from transforming_autoencoders.network.capsule import Capsule


class TransformingAutoencoder:

    def __init__(self, x, target, extra_in, input_dim, recognizer_dim, generator_dim, num_capsules):

        self.x = x
        self.target = target
        self.extra_in = extra_in

        self.num_capsules = num_capsules
        self.in_dim = input_dim
        self.recognizer_dim  = recognizer_dim
        self.generator_dim  = generator_dim

        self._inference = None
        self._loss = None
        self._summaries = []

        self.inference
        self.loss
        self.summaries

    @property
    def inference(self):
        if self._inference is None:
            capsules_out = []
            for i in range(self.num_capsules):
                with tf.variable_scope('capsule_{}'.format(i)):
                    capsule = Capsule(self.in_dim, self.recognizer_dim, self.generator_dim)
                    capsule_out = capsule.build(self.x, self.extra_in)
                    capsules_out.append(capsule_out)
            all_caps_out = tf.add_n(capsules_out)
            self._inference = tf.sigmoid(all_caps_out)
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
            x_reshaped = tf.reshape(self.x, [-1, 28, 28])
            x_pred_reshaped = tf.reshape(self.inference, [-1, 28, 28])
            target_reshaped = tf.reshape(self.target, [-1, 28, 28])

            self._summaries.append(tf.summary.scalar('autoencoder_loss', self.loss))
            self._summaries.append(tf.summary.image('input', tf.expand_dims(x_reshaped[:, :, :], -1)))
            self._summaries.append(tf.summary.image('prediction', tf.expand_dims(x_pred_reshaped[:, :, :], -1)))
            self._summaries.append(tf.summary.image('target', tf.expand_dims(target_reshaped[:, :, :], -1)))
        return self._summaries
