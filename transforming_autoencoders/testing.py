import os
import sys
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from transforming_autoencoders.utils.data_load import DataLoader


class TransformMatrix:
    """
    Transformation matrix between two views
    """
    def __init__(self, args):
        """
        Initialize `TransformMatrix` from a numpy array
        
        Parameters
        ----------
        args: Namespace
            Namespace containing experiment parameters
        """
        self.args     = args                     # experiment parameters

        self.matrix   = self._init_matrix()      # init transformation matrix
        self.n_dims   = len(self.matrix.shape)
        self.shape    = self.matrix.shape
        self.size     = self.matrix.size
        self.index_1d = 0                        # index of currently selected element

    def _init_matrix(self):
        """
        Initialize transformation matrix according to experiment parameters
        """
        if self.args.dataset == 'mnist':
            if self.args.transformation == 'translation':
                matrix = np.zeros(shape=2)
            elif self.args.transformation == 'affine':
                matrix = np.eye(3)
        else:
            raise NotImplementedError('NORB dataset still not supported in testing')
        return matrix

    def reset(self):
        """
        Reset transformation matrix to default one
        """
        self.matrix = self._init_matrix()
        self.index_1d = 0

    def update_current_value(self, delta):
        """
        Update currently selected index by delta.
        """
        self.matrix[self.index] += delta

    @property
    def index(self):
        """
        Index of element currently selected in the transformation matrix.
        """
        if self.size == 1:
            index = self.index_1d
        else:  # self.size == 2
            index = np.unravel_index(self.index_1d, dims=self.shape)
        return index

    def __repr__(self):
        """
        Represent the transformation matrix as a string
        """
        to_string = ''
        for i in range(self.size):
            to_string += 'vvvvvv  ' if i == self.index_1d else '______  '
        to_string += '\n'
        for i in range(self.size):
            index = np.unravel_index(i, dims=self.shape) if self.size > 1 else self.index_1d
            to_string += '{:06.03f}  '.format(self.matrix[index])
        return to_string


class ModelTesting:
    """
    Test a pretrained model loading weights from a checkpoint
    """
    def __init__(self, args):

        if args.restore_checkpoint is None:
            raise ValueError('Please provide a checkpoint to restore.')

        # Load data according to `args.dataset`
        self.data = DataLoader(args).load_data()

        self.restore_checkpoint = args.restore_checkpoint

        self.saver = tf.train.import_meta_graph(self.restore_checkpoint + '.meta')

        self.session = tf.Session()
        self.saver.restore(self.session, self.restore_checkpoint)

        self.cur_example_to_test = self.get_random_test_example()  # example to be tested
        self.cur_transformation  = TransformMatrix(args)

    def get_random_test_example(self):
        """
        Return a random `TransformingAutoencoderExample` from the test set.
        """
        return random.choice(self.data['test'])

    def keypress_callback(self, event):
        """
        Keypress callback allow exploration of autoencoder behavior
        """
        if event.key == 'enter':
            self.cur_example_to_test = self.get_random_test_example()  # test another image
        elif event.key == 'escape':
            self.cur_example_to_test = None
            self.cur_transformation.reset()
        elif event.key == 'right':
            self.cur_transformation.index_1d = (self.cur_transformation.index_1d + 1) % self.cur_transformation.size
        elif event.key == 'left':
            self.cur_transformation.index_1d = (self.cur_transformation.index_1d - 1) % self.cur_transformation.size
        elif event.key == 'up':
            self.cur_transformation.update_current_value(0.1)
        elif event.key == 'down':
            self.cur_transformation.update_current_value(-0.1)

        # Clear screen and show current transformation
        os.system('cls' if os.name == 'nt' else 'clear')
        print('Current transformation matrix:')
        print(self.cur_transformation)
        sys.stdout.flush()

    def get_from_graph(self, op_name):
        """
        Get output of TF operation from restored graph
        """
        return self.session.graph.get_operation_by_name(op_name).outputs[0]

    def test(self):
        """
        Test pretrained model on random examples from the test set.
        
        First, the session containing graph and pre-trained weights is loaded.
        Then, random examples from the test set are loaded and inference is performed. The result
        is visualized in a subplot which compares the input to the inference result.
        
        Using keyboard arrows, transformation matrix can be controlled. This allows the user
        to inspect how the transforming autoencoders behaves when the transformation matrix changes.
        """
        plt.ion()

        fig, axes = plt.subplots(1, 2)
        fig.canvas.mpl_connect('key_press_event', self.keypress_callback)

        with self.session as sess:

            autoencoder_inference = self.get_from_graph('autoencoder_inference')
            placeholder_input     = self.get_from_graph('placeholder_input')
            placeholder_transform = self.get_from_graph('placeholder_transformation')

            while True:

                if self.cur_example_to_test is None:
                    break

                def dummy_batch(x):
                    """ Add dummy batch dimension to input `x` """
                    return np.expand_dims(x, axis=0)

                mnist_image = self.cur_example_to_test.view_1  # random image from test set
                inference_output, = sess.run(fetches=autoencoder_inference,
                                             feed_dict={
                                                 placeholder_input:     dummy_batch(mnist_image),
                                                 placeholder_transform: dummy_batch(self.cur_transformation.matrix)
                                             })

                def prepare_for_visualization(x):
                    """ Remove dummy batch dimension from input `x` """
                    return np.squeeze(x)

                axes[0].imshow(prepare_for_visualization(mnist_image), cmap='gray')
                axes[0].set_title('Input')
                axes[1].imshow(prepare_for_visualization(inference_output), cmap='gray')
                axes[1].set_title('Output')
                plt.draw()
                plt.waitforbuttonpress()
