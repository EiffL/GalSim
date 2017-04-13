import lasagne.layers

from lasagne.layers import SliceLayer, batch_norm, ElemwiseSumLayer
from lasagne.layers import NonlinearityLayer, DenseLayer, ConcatLayer
from lasagne.nonlinearities import elu, rectify

import copy
import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams  # Limited but works on GPU
from theano.tensor.shared_randomstreams import RandomStreams

class MadeIAFLayer(lasagne.layers.MergeLayer):
    """
    IAF layer based on a MADE
    """

    def __init__(self, mu, log_sigma, h, hidden_sizes, nonlinearity=elu, **kwargs):
        """
        Initialise IAF layer
        """
        # Should define the IAF structure here and transfer all the layers
        self.mu_0 = mu
        self.log_sigma_0 = log_sigma
        self.h = h

        # Sample from the distribution
        z0 = GaussianSampleLayer(mu=mu_0, log_sigma=log_sigma_0)

        l = - sum( log_sigma + 1/2 epsilon^2 + 0.5 * log(2. * math.pi) )


        super(MadeIAFLayer, self)._init__([z, h], **kwargs)
