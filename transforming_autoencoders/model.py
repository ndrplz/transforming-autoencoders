import time
import os
import sys
import numpy as np
import tensorflow as tf
from transforming_autoencoders.trans_ae import TransformingAutoencoder


class ModelTrain:

    def __init__(self, X_trans, trans, X_original, args, resume_from_checkpoint=None):

        self.g_dim  = args.generator_dim
        self.r_dim  = args.recognizer_dim
        self.in_dim = X_trans.shape[1]
        self.num_capsules = args.num_capsules

        self.items = len(X_original)
        self.batch_size = args.batch_size
        self.steps_per_epoch = self.items // self.batch_size
        self.num_epochs = args.num_epochs

        # Optimization parameters
        self.learning_rate = args.learning_rate
        self.moving_average_decay = args.moving_average_decay

        self.X_trans = X_trans
        self.trans = trans
        self.X_original = X_original

        # Checkpoints stuff
        self.train_dir = args.train_dir
        self.resume_training = False
        if not resume_from_checkpoint:
            if tf.gfile.Exists(self.train_dir):
                tf.gfile.DeleteRecursively(self.train_dir)
            tf.gfile.MakeDirs(self.train_dir)
        else:
            self.resumeFromCheckpoint = resume_from_checkpoint
            self.resume_training = True
            print('Resuming after checkpoint: ' + resume_from_checkpoint)

        self.args = args

        print('TRAIN Directory is {}'.format(self.train_dir))

    def batch_for_step(self, step):
        return (self.X_trans[step*self.batch_size: (step+1)*self.batch_size],
                self.trans[step*self.batch_size: (step+1)*self.batch_size],
                self.X_original[step*self.batch_size: (step+1)*self.batch_size])
    
    def train(self):

        with tf.Graph().as_default():

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            opt = tf.train.AdamOptimizer(self.learning_rate)

            encoder = TransformingAutoencoder(self.in_dim, self.r_dim, self.g_dim, self.num_capsules, self.batch_size)

            # Input placeholders for each step
            X_batch_in = tf.placeholder(tf.float32, shape=[None, 784])
            X_batch_out = tf.placeholder(tf.float32, shape=[None, 784])
            extra_in = tf.placeholder(tf.float32, shape=[None, 2])

            with tf.name_scope('tower_{}'.format(0)) as scope:
                X_batch_pred = encoder.forward_pass(X_batch_in, extra_in)
                batch_loss = encoder.loss(X_batch_pred, X_batch_out)
                grads = opt.compute_gradients(batch_loss)

                tf.summary.scalar('loss', batch_loss)
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                for grad, var in grads:
                    if grad is not None:
                        if 'capsule' in var.op.name:
                            if 'capsule_0' in var.op.name:
                                print(var.op.name)
                                summaries.append(tf.summary.histogram(var.op.name + '\gradients', grad))
                        else:
                            print('no capsule- {}'.format(var.op.name))
                            summaries.append(tf.summary.histogram(var.op.name + '\gradients', grad))

                with tf.name_scope('gradients_apply'):
                    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

                # Using exponential moving average => todo Check if this works
                with tf.name_scope('exp_moving_average'):
                    variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)
                    variable_average_op = variable_averages.apply(tf.trainable_variables())

            train_op = tf.group(apply_gradient_op, variable_average_op)
            summary_op = tf.summary.merge(summaries)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
            init = tf.global_variables_initializer()

            with tf.Session() as sess:

                def count_trainable_parameters():
                    trainable_variables_shapes = [v.get_shape() for v in tf.trainable_variables()]
                    return np.sum([np.prod(s) for s in trainable_variables_shapes])
                print('Total trainable parameters: {}'.format(count_trainable_parameters()))

                sess.run(init)
                print('Variables Initialized.')

                summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
                print('Graph is saved.')

                for epoch in range(self.num_epochs):
                    start_time = time.time()
                    epoch_loss = []
                    save_summary = False
                    if epoch % self.args.save_prediction_every == 0:
                        save_summary = True

                    for step in range(self.steps_per_epoch):

                        x_batch, trans_batch, x_orig_batch = self.batch_for_step(step)
                        feed_dict = {X_batch_in: x_orig_batch, extra_in: trans_batch, X_batch_out: x_batch}

                        step_loss, _, summary = sess.run([batch_loss, train_op, summary_op], feed_dict=feed_dict)
                        if save_summary:
                            summary_writer.add_summary(summary, epoch*self.steps_per_epoch + step)
                        epoch_loss.append(step_loss)

                    epoch_loss = sum(epoch_loss)
                    duration_time = time.time() - start_time
                    print('Epoch {:d} with loss {:.3f}, ({:.3f} sec/step)'.format(epoch+1, epoch_loss, duration_time))

                    # Save model checkpoint
                    if epoch % self.args.save_checkpoint_every == 0:
                        print('Saving model checkpoint')
                        checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=epoch)

                print('Training Complete')
                sys.stdout.flush()
