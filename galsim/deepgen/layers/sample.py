import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as T
import theano


class GaussianSampleLayer(lasagne.layers.MergeLayer):
    """
    Simple Gaussian sampling layer

    Parameters
    ----------

    mean: :class:`Layer` instance
        Mean of the Gaussian distribution to sample from.

    log_var: :class:`Layer` instance
        Log variance of the Gaussian distribution to sample from.

    seed: int
        Seed for the random number generator

    """

    def __init__(self, mean, log_var,
                 seed=lasagne.random.get_rng().randint(1, 2147462579), **kwargs):
        super(GaussianSampleLayer, self).__init__([mean, log_var], **kwargs)
        self._rng = RandomStreams(seed)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
         mean, log_var = inputs
         n = self._rng.normal(mean.shape)
         z = mean + T.exp(0.5 * log_var ) * n
         return z

class GMSampleLayer(lasagne.layers.MergeLayer):
    """
    Gaussian Mixture Model sampling layer

    Parameters
    ----------

    mean: :class:`Layer` instance
        Means of the GMM, expects shape of [batch_size, dim, n_gaussians]

    log_var: :class:`Layer` instance
        Log variances of the GMM, expects shape of [batch_size, dim, n_gaussians]

    weight: :class:`Layer` instance
        Amplitudes of the GMM, expects shape of [batch_size, n_gaussians]

    seed: int
        Seed for the random number generator
    """

    def __init__(self, mean, log_var, weight,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        super(GMSampleLayer, self).__init__([mean, log_var, weight], **kwargs)
        self._rng = RandomStreams(seed)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0:2]

    def get_output_for(self, inputs, **kwargs):
        mean, log_var, weight = inputs
        batch_size, ndim, n_gaussians = mean.shape
        # Get the index of the Gaussian to sample
        w = self._rng.multinomial(n=1, pvals=weight)
        mean_w = T.batched_dot(mean, w)
        log_var_w = T.batched_dot(log_var, w)
        n = self._rng.normal(mean_w.shape)
        z = mean_w + T.exp(0.5 * log_var_w ) * n
        return z

class BernoulliSampleLayer(lasagne.layers.Layer):
    """
    Simple sampling layer drawing samples from bernoulli distributions.

    Parameters
    ----------
    mean : :class:`Layer` instances
          Parameterizing the mean value of each bernoulli distribution
    seed : int
        seed to random stream
    Methods
    ----------
    seed : Helper function to change the random seed after init is called
    """

    def __init__(self, mean,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        super(BernoulliSampleLayer, self).__init__(mean, **kwargs)

        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
        self._srng.seed(seed)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, mu, **kwargs):
        return self._srng.binomial(size=mu.shape, p=mu, dtype=mu.dtype)
