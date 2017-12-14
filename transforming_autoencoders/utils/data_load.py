from tensorflow.examples.tutorials.mnist import input_data
from transforming_autoencoders.utils.data_transform import transform_mnist_data


class DataLoader:
    """
    Load to load train, validation and test data in `ModelTraining`
    """
    def __init__(self, args):
        """
        Initialize DataLoader with command line arguments
        
        Parameters
        ----------
        args: argparse.Namespace
            Command line arguments
        """
        self.args = args

    def load_data(self):
        """
        Load appropriate data according to current dataset in use
        
        Returns
        -------
        data: dict
            Dictionary containing train, validation and test data
        """
        if self.args.dataset == 'mnist':
            # Transform and store MNIST images for each dataset split
            MNIST_data = load_MNIST_data()
            return {data_split: transform_mnist_data(x=MNIST_data[data_split],
                                                     transform_mode=self.args.transformation,
                                                     max_translation=self.args.max_translation,
                                                     sigma=self.args.sigma)
                    for data_split in ['train', 'validation', 'test']}

        elif self.args.dataset == 'norb':
            raise NotImplementedError('NORB interface still not implemented.')

        else:
            raise ValueError('{} is not a valid dataset.'.format(self.args.dataset))


def load_MNIST_data():
    """
    Load MNIST images split into train, validation and test set.

    Returns
    -------
    mnist_dict: dict
        Dictionary with keys ['train', 'validation', 'test'], 
        containing respective images data 
    """
    mnist = input_data.read_data_sets('data', one_hot=True)
    return {'train': mnist.train.images,
            'validation': mnist.validation.images,
            'test': mnist.test.images}