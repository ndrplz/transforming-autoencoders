import tensorflow as tf
from transforming_autoencoders.utils import load_train_data
from transforming_autoencoders.utils import translate
from transforming_autoencoders.model import ModelTrain


RUN = 'run_6'
TRAIN_DIR = 'trans-autoencoder-summary/'

tf.app.flags.DEFINE_string('train_dir', TRAIN_DIR+RUN, 'Directory where we write logs and checkpoints')
tf.app.flags.DEFINE_string('checkpoint_dir', TRAIN_DIR+RUN, 'Directory from where to read the checkpoint')
tf.app.flags.DEFINE_integer('num_epochs', 800, 'Number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_integer('save_checkpoint_every', 200, 'Save prediction after save_checkpoint_every epochs')
tf.app.flags.DEFINE_integer('save_pred_every', 20, 'Save prediction after save_pred_every epochs')
tf.app.flags.DEFINE_integer('save_checkpoint_after', 0, 'Save prediction after epochs')
tf.app.flags.DEFINE_integer('num_capsules', 60, 'Number of capsules')
tf.app.flags.DEFINE_integer('generator_dim', 20, 'Dimension of generator layer')
tf.app.flags.DEFINE_integer('recognizer_dim', 10, 'Dimension of recognition layer')
FLAGS = tf.app.flags.FLAGS


def main():

    train_images = load_train_data()
    X_trans, trans, X_original = translate(train_images)

    model = ModelTrain(X_trans, trans, X_original, FLAGS.num_capsules, FLAGS.recognizer_dim, FLAGS.generator_dim, X_trans.shape[1])
    model.train()


if __name__ == '__main__':
    main()
