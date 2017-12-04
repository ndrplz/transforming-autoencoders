# capsules

Getting hands dirty with Hinton's capsules. In TensorFlow. 

## [Transforming Autoencoders](https://github.com/ndrplz/capsules/tree/master/transforming_autoencoders)

TensorFlow implementation of the folowing [paper](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf).

> Hinton, Geoffrey E., Alex Krizhevsky, and Sida D. Wang. "Transforming auto-encoders." International Conference on Artificial Neural Networks. Springer, Berlin, Heidelberg, 2011.

In this paper a simple *capsule-based network* is used to model different viewing conditions of an implicitly defined visual entity. Each capsule outputs both the probability that a particular visual entity is present and a set of *instantiation parameters* like pose, lighting and deformation of the visual entity relative to a canonical version of that entity. 

The recognition probablity is multiplied elementwise to the capsule output. Thus, the less confident the capsule is that the visual entity is present in its limited domain the less the output of that capsule will be weighted in the overall autoencoder prediction.

In pooling-based CNNs, activations are *invariant* (*i.e.* do not change) for small pose variations of the target visual entity. Conversely, in a trained Transforming Autoencoder the probability of visual entity is expected to be invariant as the entity moves over the manifold of possible appearances, while instantiation parameters are *equivariant* – as the viewing conditions change and the entity moves over the appearance manifold, the instantiation parameters change by a corresponding amount because they are representing the intrinsic coordinates of the entity on the appearance manifold.

<p align="center"><img src="https://raw.githubusercontent.com/nikhil-dce/Transforming-Autoencoder-TF/master/extras/architecture.png" width="800"></p>

### Usage
Still TODO

### Limitations

* Up to this moment, only experiment on translations of MNIST digits is implemented.

---

### Acknowledgements

* First template for Transforming Autoencoders implementation was taken from [this](https://github.com/nikhil-dce/Transforming-Autoencoder-TF) repository by [nikhil-dce](https://github.com/nikhil-dce).
