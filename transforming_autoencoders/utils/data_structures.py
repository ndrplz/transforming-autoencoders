import numpy as np


class TransformingAutoencoderExample:
    """
    Class that models a single example for a Transforming Autoencoder.

    Each example is defined by three things:
        - First view (image pre-transformation)
        - Second view (image post-transformation)
        - Transformation applied to go from the 1st to the 2nd view
    """

    def __init__(self, view_1, view_2, transformation):
        self.view_1 = view_1
        self.view_2 = view_2
        self.transformation = transformation

    def show(self, subplots):
        """
        Display TransformingAutoencoderExample on matplotlib subplot

        Parameters
        ----------
        subplots: (fig, axes)
            Return value from `matplotlib.pyplot.subplots(1, 2)`

        Returns
        -------
        None
        """
        fig, axes = subplots
        fig.suptitle('Transformation: {}'.format(self.transformation))
        axes[0].imshow(np.squeeze(self.view_1), cmap='gray')
        axes[1].imshow(np.squeeze(self.view_2), cmap='gray')
