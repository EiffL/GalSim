import math

import lasagne

import theano.tensor as T
import theano

from lasagne.layers import get_output_shape

from ..distributions import log_normal2, log_gm2

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

        invsig2_t = T.exp( - t_logvar )
        invsig2_d = T.exp( - d_logvar )

        q_mu = (t_mu*invsig2_t + d_mu*invsig2_d)/(invsig2_t + invsig2_d)
        return q_mu

class MergeLogVarLayer(lasagne.layers.MergeLayer):

    def __init__(self, d_logvar, t_logvar, **kwargs):
        super(MergeLogVarLayer, self).__init__([d_logvar, t_logvar], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        d_logvar, t_logvar = inputs

        invsig2_t = T.exp(- t_logvar)
        invsig2_d = T.exp(- d_logvar)

        return  - T.log(invsig2_t + invsig2_d)


class LogNormalLayer(lasagne.layers.MergeLayer):
    """
    Computes the element wise log likelihood with a Gaussian model
    """
    
    def __init__(self, z, mean, log_var=None, cst_std=1, epsilon=1e-7, **kwargs):
        if log_var is None:
            super(LogNormalLayer, self).__init__([z,mean], **kwargs)
            self.cst_std = cst_std
        else:
            super(LogNormalLayer, self).__init__([z,mean,log_var], **kwargs)
            self.cst_std = None
        self.epsilon = epsilon
        self.in_shape = get_output_shape(z)
        
    def get_output_shape_for(self, input_shapes):
        return [input_shapes[0][0]]
    
    def get_output_for(self, inputs, **kwargs):
        if self.cst_std is None:
            z, mean, logvar = inputs
            pz = log_normal2(z, mean, logvar, eps=self.epsilon)
        else:
            z, mean = inputs
            c = - 0.5 * math.log(2*math.pi)
            pz = c - math.log(self.cst_std) - (z - mean)**2 / (2 * self.cst_std**2)
        return pz.sum(axis=range(1,len(self.in_shape)))


class LogGMLayer(lasagne.layers.MergeLayer):
    """
    Computes the element wise log likelihood with a Gaussian mixture model
    """
    
    def __init__(self, z, mean, log_var, weight, epsilon=1e-7, **kwargs):
        super(LogGMLayer, self).__init__([z,mean,log_var, weight], **kwargs)
        self.epsilon = epsilon
        
    def get_output_shape_for(self, input_shapes):
        return [input_shapes[0][0]]
    
    def get_output_for(self, inputs, **kwargs):
        z, mean, log_var, w = inputs
        return log_gm2(z, mean, log_var, w, eps=self.epsilon)

class KLLayer(lasagne.layers.MergeLayer):
    """
    Merges log likelihoods from prior and posterior into a KL divergence
    """
        
    def __init__(self, log_pz, log_qz, negative=True, **kwargs):
        super(KLLayer, self).__init__([log_pz, log_qz], **kwargs)
        self.negative = negative
        
    def get_output_shape_for(self, input_shapes):
        return [input_shapes[0]]
    
    def get_output_for(self, inputs, **kwargs):
        log_pz, log_qz = inputs
        kl = T.nnet.relu(log_qz - log_pz)
        if self.negative:
            return - kl
        else:
            return kl
    
