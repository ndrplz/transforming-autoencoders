import os
import argparse
from time import time
from os.path import join
from transforming_autoencoders.training import ModelTraining
from transforming_autoencoders.testing import ModelTesting


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'test'], help='Choose mode (`train` or `test`)')
    parser.add_argument('--transformation', choices=['translation', 'affine'], default='affine', help='Transformation')
    parser.add_argument('-n', '--num_capsules', type=int, default=10, help='Number of capsules')
    parser.add_argument('-g', '--generator_dim', type=int, default=30, help='Dimension (neurons) of generator layer')
    parser.add_argument('-r', '--recognizer_dim', type=int, default=30, help='Dimension (neurons) of recognition layer')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'norb'], help='Dataset to use.')
    parser.add_argument('--train_dir', type=str, default=join('checkpoints', str(time())), help='Checkpoints directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_checkpoint_every', type=int, default=10, help='Epochs between saved checkpoints')
    parser.add_argument('--save_prediction_every', type=int, default=10, help='Epochs between saved predictions')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='Moving average decay')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of Adam optimizer')
    parser.add_argument('--max_translation', type=int, default=5, help='Max data translation allowed')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma parametrizing affine transformations')
    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use in case of multiple GPUs (default=0)')
    parser.add_argument('--restore_checkpoint', type=str, help='Path to restore checkpoint (in `test` mode)')
    return parser.parse_args()


if __name__ == '__main__':

    # Parse command line arguments
    command_line_arguments = parse_arguments()

    # Appropriately set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(command_line_arguments.gpu)

    if command_line_arguments.mode == 'train':
        # Start training
        training = ModelTraining(command_line_arguments)
        training.train()
    elif command_line_arguments.mode == 'test':
        # Start testing
        testing = ModelTesting(command_line_arguments)
        testing.test()
