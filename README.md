# Transforming-Autoencoders

TensorFlow implementation of the following [paper](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf).

> Hinton, Geoffrey E., Alex Krizhevsky, and Sida D. Wang. "Transforming auto-encoders." International Conference on Artificial Neural Networks. Springer, Berlin, Heidelberg, 2011.

In this paper a simple *capsule-based network* is used to model different viewing conditions of an implicitly defined visual entity. Each capsule outputs both the probability that a particular visual entity is present and a set of *instantiation parameters* like pose, lighting and deformation of the visual entity relative to a canonical version of that entity. 

The recognition probablity is multiplied elementwise to the capsule output. Thus, the less confident the capsule is that the visual entity is present in its limited domain the less the output of that capsule will be weighted in the overall autoencoder prediction.

In pooling-based CNNs, activations are *invariant* (*i.e.* do not change) for small pose variations of the target visual entity. Conversely, in a trained Transforming Autoencoder the probability of visual entity is expected to be invariant as the entity moves over the manifold of possible appearances, while instantiation parameters are *equivariant* – as the viewing conditions change and the entity moves over the appearance manifold, the instantiation parameters change by a corresponding amount because they are representing the intrinsic coordinates of the entity on the appearance manifold.

## Usage

````
usage: main.py [-h] [--transformation {translation,affine}] [-n NUM_CAPSULES]
               [-g GENERATOR_DIM] [-r RECOGNIZER_DIM] [--dataset {mnist,norb}]
               [--train_dir TRAIN_DIR] [--num_epochs NUM_EPOCHS]
               [--batch_size BATCH_SIZE]
               [--save_checkpoint_every SAVE_CHECKPOINT_EVERY]
               [--save_prediction_every SAVE_PREDICTION_EVERY]
               [--moving_average_decay MOVING_AVERAGE_DECAY]
               [--learning_rate LEARNING_RATE]
               [--max_translation MAX_TRANSLATION] [--sigma SIGMA] [--gpu GPU]
               [--restore_checkpoint RESTORE_CHECKPOINT]
               {train,test}

positional arguments:
  {train,test}          Choose mode (`train` or `test`)

optional arguments:
  -h, --help            show this help message and exit
  --transformation {translation,affine}
                        Transformation
  -n NUM_CAPSULES, --num_capsules NUM_CAPSULES
                        Number of capsules
  -g GENERATOR_DIM, --generator_dim GENERATOR_DIM
                        Dimension (neurons) of generator layer
  -r RECOGNIZER_DIM, --recognizer_dim RECOGNIZER_DIM
                        Dimension (neurons) of recognition layer
  --dataset {mnist,norb}
                        Dataset to use.
  --train_dir TRAIN_DIR
                        Checkpoints directory
  --num_epochs NUM_EPOCHS
                        Number of training epochs
  --batch_size BATCH_SIZE
                        Batch size
  --save_checkpoint_every SAVE_CHECKPOINT_EVERY
                        Epochs between saved checkpoints
  --save_prediction_every SAVE_PREDICTION_EVERY
                        Epochs between saved predictions
  --moving_average_decay MOVING_AVERAGE_DECAY
                        Moving average decay
  --learning_rate LEARNING_RATE
                        Learning rate of Adam optimizer
  --max_translation MAX_TRANSLATION
                        Max data translation allowed
  --sigma SIGMA         Sigma parametrizing affine transformations
  --gpu GPU             Which GPU to use in case of multiple GPUs (default=0)
  --restore_checkpoint RESTORE_CHECKPOINT
                        Path to restore checkpoint (in `test` mode)

````

### Code

Transforming Autoencoder implementation and more detailed code structure description can be found in [`transforming_autoencoders/`](https://github.com/ndrplz/capsules/tree/master/transforming_autoencoders)
