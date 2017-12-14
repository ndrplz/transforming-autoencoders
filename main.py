import argparse
from time import time
from os.path import join
from transforming_autoencoders.training import ModelTraining


def parse_arguments():
    parser = argparse.ArgumentParser(description=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('num_capsules', type=int,
                        help='Number of capsules')
    parser.add_argument('generator_dim', type=int,
                        help='Dimension (number of neurons) of generator layer')
    parser.add_argument('recognizer_dim', type=int,
                        help='Dimension (number of neurons) of recognition layer')
    parser.add_argument('transformation', type=str,
                        help='Transformation applied (translation, affine)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'norb'],
                        help='Which dataset to use (MNIST or NORB)')
    parser.add_argument('--train_dir', type=str, default=join('checkpoints', str(time())), metavar='',
                        help='Checkpoints directory')
    parser.add_argument('--num_epochs', type=int, default=100, metavar='',
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, metavar='',
                        help='Batch size')
    parser.add_argument('--save_checkpoint_every', type=int, default=10, metavar='',
                        help='Epochs between saved checkpoints')
    parser.add_argument('--save_prediction_every', type=int, default=10, metavar='',
                        help='Epochs between saved predictions')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, metavar='',
                        help='Moving average decay')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='',
                        help='Learning rate of Adam optimizer')
    parser.add_argument('--max_translation', type=int, default=5, metavar='',
                        help='Max data translation allowed')
    parser.add_argument('--sigma', type=float, default=0.1, metavar='',
                        help='Sigma parametrizing affine transformations')
    return parser.parse_args()


if __name__ == '__main__':

    # Parse command line arguments
    command_line_arguments = parse_arguments()

    # Start training
    training = ModelTraining(command_line_arguments)
    training.train()
