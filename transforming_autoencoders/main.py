import argparse
from time import time
from os.path import join
from transforming_autoencoders.utils.data_handling import load_train_data
from transforming_autoencoders.utils.data_handling import translate_randomly
from transforming_autoencoders.training import ModelTraining


def parse_arguments():
    parser = argparse.ArgumentParser(description=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dir', type=str, default=join('checkpoints', str(time())), help='Checkpoints directory')
    parser.add_argument('--num_capsules', type=int, default=50, help='Number of capsules')
    parser.add_argument('--generator_dim', type=int, default=20, help='Dimension of generator layer')
    parser.add_argument('--recognizer_dim', type=int, default=10, help='Dimension of recognition layer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_checkpoint_every', type=int, default=10, help='Epochs between saved checkpoints')
    parser.add_argument('--save_prediction_every', type=int, default=10, help='Epochs between saved predictions')
    parser.add_argument('--moving_average_decay', type=float, default=0.9999, help='Moving average decay')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate of Adam optimizer')

    return parser.parse_args()


def main():

    # Parse command line arguments
    args = parse_arguments()

    # Load and preprocess MNIST data
    train_images = load_train_data()
    X_trans, trans, X_original = translate_randomly(train_images, max_offset=5)

    # Start training
    model = ModelTraining(X_trans, trans, X_original, args)
    model.train()


if __name__ == '__main__':
    main()
