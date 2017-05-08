import math

import lasagne

import theano.tensor as T
import theano

from lasagne.layers import get_output_shape

from ..distributions import log_normal2, log_gm2, log_bernoulli

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
        super(self.__class__, self).__init__([d_mu, d_logvar, t_mu, t_logvar], **kwargs)

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
        super(self.__class__, self).__init__([d_logvar, t_logvar], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        d_logvar, t_logvar = inputs

        invsig2_t = T.exp(- t_logvar)
        invsig2_d = T.exp(- d_logvar)

        return  - T.log(invsig2_t + invsig2_d)

class BernoulliLikelihoodLayer(lasagne.layers.MergeLayer):
    """
    Computes the log likelihood with a Bernoulli model
    """
    def __init__(self, z, mean,  epsilon=0, **kwargs):
        super(self.__class__, self).__init__([z,mean], **kwargs)
        self.epsilon = epsilon
        self.in_shape = get_output_shape(z)
        self.in_logvar_shape = get_output_shape(log_var)
        
    def get_output_shape_for(self, input_shapes):
        return [input_shapes[0][0]]
    
    def get_output_for(self, inputs, **kwargs):
        z, mean = inputs
        pz = log_bernoulli(z, mean, eps=self.epsilon)
        return pz.sum(axis=range(1,len(self.in_shape)))
 
class GaussianLikelihoodLayer(lasagne.layers.MergeLayer):
    """
    Computes the log likelihood with a Gaussian model
    """
    
    def __init__(self, z, mean, log_var, epsilon=1e-7, **kwargs):
        super(self.__class__, self).__init__([z,mean,log_var], **kwargs)
        self.epsilon = epsilon
        self.in_shape = get_output_shape(z)
        self.in_logvar_shape = get_output_shape(log_var)
        
    def get_output_shape_for(self, input_shapes):
        return [input_shapes[0][0]]
    
    def get_output_for(self, inputs, **kwargs):
        z, mean, logvar = inputs
        
        # First make sure all dimensions with ones are broadcastable
        dims = []
        for i,s in enumerate(self.in_logvar_shape):
            if s == 1:
                dims.append(i)
        if len(dims) >0:
            logvar = T.addbroadcast(logvar, *dims)
        
        # If logvar is a diagonal, pad it to the right until we match the dimension of x
        if len(self.in_logvar_shape) < len(self.in_shape):                                  
            logvar = T.shape_padright(logvar, n_ones=(len(self.in_shape) - len(self.in_logvar_shape)))

        pz = log_normal2(z, mean, logvar, eps=self.epsilon)
        return pz.sum(axis=range(1,len(self.in_shape)))

class FourierGaussianLikelihoodLayer(lasagne.layers.MergeLayer):
    """
    Computes the log likelihood with a Gaussian model
    """
    
    def __init__(self, z, mean, log_var, epsilon=1e-7, log_var_max=5, normalise=True, **kwargs):
        super(self.__class__, self).__init__([z,mean,log_var], **kwargs)
        self.epsilon = epsilon
        self.log_var_max = log_var_max
        self.in_shape = get_output_shape(z)
        self.normalise = normalise
        
    def get_output_shape_for(self, input_shapes):
        return [input_shapes[0][0]]
    
    def get_output_for(self, inputs, **kwargs):
        z, mean, log_var = inputs
        
        
        # Ok, here we assume the logvar to have the same shape as our images
        pz = - ((z - mean)**2).sum(axis=-1) / (2 * T.exp(log_var) + self.epsilon)
        if self.normalise:
            c = - 0.5 * math.log(2*math.pi)
            pz = pz + c - log_var/2
        
        # Exclude from likelihood points where the variance is essentially 0
        pz = T.where(log_var >= self.log_var_max, 0, pz)
        
        return pz.sum(axis=range(1,len(self.in_shape[:-1])))


class GMLikelihoodLayer(lasagne.layers.MergeLayer):
    """
    Computes the elementwise log likelihood with a Gaussian mixture model
    """
    
    def __init__(self, z, mean, log_var, weight, epsilon=1e-7, **kwargs):
        super(self.__class__, self).__init__([z,mean,log_var, weight], **kwargs)
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
        
    def __init__(self, log_pz, log_qz, negative=True, clip_negative=True, **kwargs):
        super(self.__class__, self).__init__([log_pz, log_qz], **kwargs)
        self.negative = negative
        self.clip_negative = clip_negative
        
    def get_output_shape_for(self, input_shapes):
        return [input_shapes[0]]
    
    def get_output_for(self, inputs, **kwargs):
        log_pz, log_qz = inputs
        if self.clip_negative:
            kl = T.nnet.relu(log_qz - log_pz)
        else:
            kl = log_qz - log_pz
        if self.negative:
            return - kl
        else:
            return kl
    
