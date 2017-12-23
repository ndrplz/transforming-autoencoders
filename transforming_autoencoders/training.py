import numpy as np
import tensorflow as tf
from tqdm import tqdm
from os.path import join
from transforming_autoencoders.network.transforming_autoencoder import TransformingAutoencoder
from transforming_autoencoders.utils.data_load import DataLoader


class ModelTraining:

    def __init__(self, args):

        # Load data according to `args.dataset`
        self.data = DataLoader(args).load_data()

        # Hyper-parameters
        self.input_shape    = self.data['train'][0].view_1.shape
        self.generator_dim  = args.generator_dim
        self.recognizer_dim = args.recognizer_dim
        self.num_capsules   = args.num_capsules
        self.transformation = args.transformation

        # Epoch parameters
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.steps_per_epoch = {data_split: len(self.data[data_split]) // self.batch_size
                                for data_split in ['train', 'validation', 'test']}

        # Optimization parameters
        self.learning_rate = args.learning_rate
        self.moving_average_decay = args.moving_average_decay

        # Checkpoints
        self.train_dir = args.train_dir
        print('Checkpoint directory: {}'.format(self.train_dir))

        # Summary writers
        self.writers = {
            'train': tf.summary.FileWriter(join(self.train_dir, 'train')),
            'validation': tf.summary.FileWriter(join(self.train_dir, 'validation'))
        }

        self.args = args

    def batch_for_step(self, data_split, step):
        dataset_batch = self.data[data_split][step * self.batch_size: (step + 1) * self.batch_size]
        return ([x.view_1 for x in dataset_batch],
                [x.view_2 for x in dataset_batch],
                [x.transformation for x in dataset_batch])

    def random_batch(self, data_split):
        random_indexes = np.random.randint(0, len(self.data[data_split]), size=self.batch_size)
        random_batch   = [self.data[data_split][i] for i in random_indexes]
        return ([x.view_1 for x in random_batch],
                [x.view_2 for x in random_batch],
                [x.transformation for x in random_batch])

    def should_save_predictions(self, epoch):
        return epoch % self.args.save_prediction_every == 0

    def should_save_checkpoints(self, epoch):
        return epoch % self.args.save_checkpoint_every == 0

    def train(self):
        with tf.Graph().as_default():

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            opt = tf.train.AdamOptimizer(self.learning_rate)

            # Placeholders
            autoencoder_input  = tf.placeholder(tf.float32, shape=[None] + list(self.input_shape),
                                                name='placeholder_input')
            autoencoder_target = tf.placeholder(tf.float32, shape=[None] + list(self.input_shape),
                                                name='placeholder_target')

            # Extra input placeholder has different shape according to transformation applied
            if self.args.transformation == 'translation':
                extra_input_shape = [None, 2]
            elif self.args.transformation == 'affine':
                extra_input_shape = [None, 3, 3]
            extra_input = tf.placeholder(tf.float32, shape=extra_input_shape, name='placeholder_transformation')

            # Transforming autoencoder model
            autoencoder = TransformingAutoencoder(x=autoencoder_input, target=autoencoder_target,
                                                  extra_input=extra_input, input_shape=self.input_shape,
                                                  recognizer_dim=self.recognizer_dim, generator_dim=self.generator_dim,
                                                  num_capsules=self.num_capsules, transformation=self.transformation)

            with tf.name_scope('tower_{}'.format(0)) as scope:

                gradients = opt.compute_gradients(autoencoder.loss)

                with tf.name_scope('gradients_apply'):
                    apply_gradient_op = opt.apply_gradients(gradients, global_step=global_step)

                # Using exponential moving average
                with tf.name_scope('exp_moving_average'):
                    variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)
                    variable_average_op = variable_averages.apply(tf.trainable_variables())

            train_op = tf.group(apply_gradient_op, variable_average_op)

            scalar_summary_op = tf.summary.merge([s for s in autoencoder.summaries if 'autoencoder_loss' in s.name])
            image_summary_op  = tf.summary.merge([s for s in autoencoder.summaries if 'autoencoder_loss' not in s.name])

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                # Display the number of trainable parameters
                def count_trainable_parameters():
                    trainable_variables_shapes = [v.get_shape() for v in tf.trainable_variables()]
                    return np.sum([np.prod(s) for s in trainable_variables_shapes])
                print('Total trainable parameters: {}'.format(count_trainable_parameters()))

                # Training loop
                for epoch in range(self.num_epochs):

                    epoch_loss = []

                    for step in tqdm(range(self.steps_per_epoch['train'])):

                        global_step = epoch * self.steps_per_epoch['train'] + step

                        # Perform one training step
                        x_view_1_batch, x_view_2_batch, trans_batch = self.batch_for_step('train', step)
                        step_loss, _, loss_summary = sess.run(fetches=[autoencoder.loss, train_op, scalar_summary_op],
                                                              feed_dict={autoencoder_input:  x_view_1_batch,
                                                                         autoencoder_target: x_view_2_batch,
                                                                         extra_input: trans_batch})
                        self.writers['train'].add_summary(loss_summary, global_step)

                        epoch_loss.append(step_loss)

                        # Save loss summary on validation set
                        x_view_1_batch, x_view_2_batch, trans_batch = self.random_batch('validation')
                        loss_summary, = sess.run(fetches=[scalar_summary_op],
                                                 feed_dict={autoencoder_input: x_view_1_batch,
                                                            autoencoder_target: x_view_2_batch,
                                                            extra_input: trans_batch})
                        self.writers['validation'].add_summary(loss_summary, global_step)

                    print('Epoch {:03d} - average training loss: {:.2f}'.format(epoch+1, np.mean(epoch_loss)))

                    if self.should_save_predictions(epoch):
                        for data_split in ['train', 'validation']:
                            print('Saving predictions on {} set...'.format(data_split))
                            x_view_1_batch, x_view_2_batch, trans_batch = self.random_batch(data_split)
                            image_summary = sess.run(fetches=image_summary_op,
                                                     feed_dict={autoencoder_input:  x_view_1_batch,
                                                                autoencoder_target: x_view_2_batch,
                                                                extra_input: trans_batch})
                            self.writers[data_split].add_summary(image_summary, global_step)

                    if self.should_save_checkpoints(epoch):
                        print('Saving checkpoints...')
                        saver.save(sess, join(self.train_dir, 'model.ckpt'), global_step=epoch)
