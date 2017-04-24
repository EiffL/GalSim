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
