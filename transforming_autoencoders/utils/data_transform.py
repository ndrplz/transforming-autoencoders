import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import AffineTransform
from skimage.transform import warp
from transforming_autoencoders.utils.data_structures import TransformingAutoencoderExample


def get_random_affine_matrix(sigma, max_translation):
    """
    Get a random affine transformation matrix

    Parameters
    ----------
    sigma: float
        Parametrizes the gaussian noise added to transformation matrix
    max_translation:
        Maximum translation allowed in the transformation matrix

    Returns
    -------
    matrix: ndarray
        3x3 matrix representing a random affine transformation
    """
    R = np.eye(2) + sigma * np.random.normal(0, 1, size=[2, 2])            # rotation and scale
    T = np.random.uniform(-max_translation, max_translation, size=[2, 1])  # translation
    H = np.array([[0., 0., 1.]])                                           # homogeneous part
    return np.concatenate([np.concatenate([R, T], axis=1), H], axis=0)


def transform_mnist_data(x, transform_mode, max_translation=5, sigma=0.1, show=False):
    """
    Transform MNIST data to generate appropriate data for transforming autoencoder training.

    Parameters
    ----------
    x: ndarray
        MNIST images
    transform_mode: str
        Transformation in ['translation', 'affine']
    max_translation: int
        Maximum translation allowed in the transformation matrix
    sigma: float
        Parametrizes the gaussian noise added to transformation matrix
    show: bool
        If True both original and transformed images are shown

    Returns
    -------
    data: dict
        Dictionary containing both original and transformed data
        along with transformations applied.
    """
    if transform_mode not in ['translation', 'affine']:
        raise ValueError('Mode "{}" not supported.'.format(transform_mode))

    if show:
        plt.ion()
        _, [ax1, ax2] = plt.subplots(1, 2)

    examples = []

    for i in np.random.permutation(len(x)):  # notice shuffling here

        mnist_image = np.reshape(x[i], (28, 28))  # MNIST is distributed flattened

        if transform_mode == 'translation':
            translation_x  = random.randint(-max_translation, max_translation)
            translation_y  = random.randint(-max_translation, max_translation)
            transformation = [translation_x, translation_y]
            transformed_image = np.roll(np.roll(mnist_image, translation_x, axis=0), translation_y, axis=1)

        else:  # transform_mode == 'affine':
            transformation = get_random_affine_matrix(sigma=sigma, max_translation=max_translation)
            transformed_image = warp(mnist_image, AffineTransform(matrix=transformation))

        if show:
            ax1.imshow(mnist_image)
            ax2.imshow(transformed_image)
            plt.show()
            plt.waitforbuttonpress()

        example = TransformingAutoencoderExample(view_1=mnist_image,
                                                 view_2=transformed_image,
                                                 transformation=transformation)
        examples.append(example)

    return examples
