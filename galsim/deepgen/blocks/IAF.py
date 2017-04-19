import lasagne
import lasagne.layers

from lasagne.layers import batch_norm, ElemwiseSumLayer
from lasagne.layers import NonlinearityLayer, DenseLayer, ConcatLayer
from lasagne.layers import get_output, get_output_shape
from lasagne.nonlinearities import elu, rectify, sigmoid, tanh, identity
from lasagne.layers import BatchNormLayer, NonlinearityLayer

from ..layers.transform import ClampLogVarLayer, ScaleShiftLayer

from .MADE import MADE
from ..layers.sample import GaussianSampleLayer

import theano
import theano.tensor as T

import math

class MergeIAFLayer(lasagne.layers.MergeLayer):
    """
    This layer applies the IAF transform
    """

    def __init__(self, z, log_var, m, **kwargs):
        """
        Initialisation
        """
        super(MergeIAFLayer, self).__init__([z ,log_var, m], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        z, log_var, m = inputs
        sigma = T.exp(0.5*log_var)
        return z * sigma + (1 - sigma) * m
    

def MADE_IAF(input_net, code_size, hidden_sizes, inputs, apply_nonlinearity=False):
    """
    Block defining an IAF scheme based on multi-layer MADE transforms

    Returns the IAF code z, and the scalar value of log(q(z | x))
    """
    
    if apply_nonlinearity:
        input_net = NonlinearityLayer(BatchNormLayer(input_net), elu)
    
    mu = DenseLayer(input_net, num_units=code_size, nonlinearity=None)
    log_var= ClampLogVarLayer(DenseLayer(input_net, num_units=code_size, nonlinearity=None))

    # Sample from the Gaussian distribution
    z0 = GaussianSampleLayer(mean=mu, log_var=log_var, name='z0_sample')

    # Initializes the value of log(q(z | x))
    m, ls, eps = get_output([mu, log_var, z0], inputs=inputs)
    l = - 0.5 * T.sum( ls + (eps - m)**2 / ( T.exp(ls) +1e-5)  + math.log(2. * math.pi), axis=-1)

    z = z0

    # Add the MADE layers
    for i,h_size in enumerate(hidden_sizes):
        m = MADE(z, h_size, output_nonlinearity=None).reset('Full', i).get_output_layer()
        # log_var = MADE(z, h_size, output_nonlinearity=tanh).reset('Full',i).get_output_layer()

        z = ElemwiseSumLayer([z, m]) # MergeIAFLayer(z, log_var, m)

        #s = get_output(log_var, inputs=inputs)

        #l = l - T.sum(0.5*s, axis=-1)

    return z, l
