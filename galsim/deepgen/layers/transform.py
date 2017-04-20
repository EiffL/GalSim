import lasagne

import theano.tensor as T
import theano

class ScaleShiftLayer(lasagne.layers.Layer):
    """
    Simple scaling and shifting layer with fixed parameters.

    Parameters
    ----------
    scale : float, optional
        scaling parameter. (default: 1.0)
    shift : float, optional
        shift parameter. (default: 0.0)

    """

    def __init__(self, input, scale=1.0, shift=0.0,
                 **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)
        self._scale = scale
        self._shift = shift

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, X, **kwargs):
        return  self._scale * ( self._shift + X)


class CropLayer(lasagne.layers.Layer):
    """
    Simple scaling and shifting layer with fixed parameters.

    Parameters
    ----------
    scale : float, optional
        scaling parameter. (default: 1.0)
    shift : float, optional
        shift parameter. (default: 0.0)

    """

    def __init__(self, input, size,
                 **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)
        self._size = size

    def get_output_shape_for(self, input_shape):
        shape = (input_shape[0], input_shape[1], self._size, self._size)
        return shape

    def get_output_for(self, X, **kwargs):
        s = X.shape
        sx = s[-1]/2 - 1
        return  X[:,:,sx - self._size/2: sx + self._size/2,
                      sx - self._size/2: sx + self._size/2]

class ClampLogVarLayer(lasagne.layers.Layer):
    """
    Simple transform applied to logvar to keep it within reasonable
    range

    Parameters
    ----------
    scale : float, optional
        scaling parameter. (default: 1.0)
    shift : float, optional
        shift parameter. (default: 0.0)

    """

    def __init__(self, input, scale=1.0, shift=1.0001, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)
        self._scale = scale
        self._shift = shift

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, X, **kwargs):
        var = self._scale * (self._shift + T.tanh(X) )
        return T.log(var)

class MergeMeanLayer(lasagne.layers.MergeLayer):

    def __init__(self, d_mu, d_logvar, t_mu, t_logvar, **kwargs):
        super(MergeMeanLayer, self).__init__([d_mu, d_logvar, t_mu, t_logvar], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):

        d_mu, d_logvar, t_mu, t_logvar = inputs

        invsig2_t = T.exp(- 2. * t_logvar)
        invsig2_d = T.exp(- 2. * d_logvar)

        q_mu = (t_mu*invsig2_t + d_mu*invsig2_d)/(invsig2_t + invsig2_d)
        return q_mu

class MergeLogVarLayer(lasagne.layers.MergeLayer):

    def __init__(self, d_logvar, t_logvar, **kwargs):
        super(MergeLogVarLayer, self).__init__([d_logvar, t_logvar], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        d_logvar, t_logvar = inputs

        invsig2_t = T.exp(- 2. * t_logvar)
        invsig2_d = T.exp(- 2. * d_logvar)

        return  - T.log(invsig2_t + invsig2_d)
