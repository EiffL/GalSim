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

class RFFTLayer(lasagne.layers.Layer):
    """
    Applies a Fourier Transform to the two trailing dimensions
    WARNING: square images, even size
    Expects images of the form [batch, nc, n_x, n_x]
    """

    def __init__(self, input, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

    def get_output_shape_for(self, input_shape):
        shape = input_shape[:-1]
        shape += (input_shape[-1]//2 + 1,)
        shape += (2,)
        return shape

    def get_output_for(self, X, **kwargs):
        s = X.shape
        X = X.reshape((s[0]*s[1],s[2], s[3]))
        Xt = T.fft.rfft(X, norm='ortho')
        snew = Xt.shape
        return  Xt.reshape((s[0], s[1], snew[1], snew[2], 2))


class iRFFTLayer(lasagne.layers.Layer):
    """
    Computes inverse Fourier Transform
    WARNING: square images, even size
    Expects images of the form [batch, nc, n_x, n_x/2 + 1, 2]
    """
    
    def __init__(self, input, **kwargs):
        super(self.__class__, self).__init__(input, **kwargs)

    def get_output_shape_for(self, input_shape):
        s = input_shape
        out_shape = [s[0], s[1], s[2], s[2]]
        return out_shape

    def get_output_for(self, X, **kwargs):
        s = X.shape
        X2 = X.reshape((s[0]*s[1],s[2], s[3], 2))
        Xt = T.fft.irfft(X2, norm='ortho')
        snew = Xt.shape
        return  Xt.reshape((s[0], s[1], snew[1], snew[2]))
    

# "Borrowed" from https://github.com/ajbrock/Neural-Photo-Editor/blob/master/layers.py
# Subpixel Upsample Layer from (https://arxiv.org/abs/1609.05158)
# This layer uses a set of r^2 set_subtensor calls to reorganize the tensor in a subpixel-layer upscaling style
# as done in the ESPCN Magic ony paper for super-resolution.
# r is the upscale factor.
# c is the number of output channels.
class SubpixelLayer(lasagne.layers.Layer):
    def __init__(self, incoming, r, c, **kwargs):
        super(self.__class__, self).__init__(incoming, **kwargs)
        self.r=r # Upscale factor
        self.c=c # number of output channels
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0],self.c,self.r*input_shape[2],self.r*input_shape[3])

    def get_output_for(self, input, deterministic=False, **kwargs):
        out = T.zeros((input.shape[0],self.output_shape[1],self.output_shape[2],self.output_shape[3]))
        for x in xrange(self.r): # loop across all feature maps belonging to this channel
            for y in xrange(self.r):
                out=T.set_subtensor(out[:,:,x::self.r,y::self.r],input[:, self.r*x+y::self.r*self.r,:,:])
        return out
