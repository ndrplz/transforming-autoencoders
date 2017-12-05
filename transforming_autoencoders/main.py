import argparse
from time import time
from os.path import join
from transforming_autoencoders.training import ModelTraining


def parse_arguments():
    parser = argparse.ArgumentParser(description=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('num_capsules', type=int, help='Number of capsules')
    parser.add_argument('generator_dim', type=int, help='Dimension of generator layer')
    parser.add_argument('recognizer_dim', type=int, help='Dimension of recognition layer')
    parser.add_argument('transformation', type=str, help='Transformation applied (translation, affine)')
    parser.add_argument('--train_dir', type=str, default=join('checkpoints', str(time())), help='Checkpoints directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs', metavar='')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size', metavar='')
    parser.add_argument('--save_checkpoint_every', type=int, default=10, help='Epochs between saved checkpoints', metavar='')
    parser.add_argument('--save_prediction_every', type=int, default=10, help='Epochs between saved predictions', metavar='')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='Moving average decay', metavar='')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of Adam optimizer', metavar='')
    parser.add_argument('--max_translation', type=int, default=5, help='Max data translation allowed', metavar='')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma parametrizing affine transformations', metavar='')

    return parser.parse_args()


if __name__ == '__main__':

    # Parse command line arguments
    command_line_arguments = parse_arguments()

    # Start training
    training = ModelTraining(command_line_arguments)
    training.train()
