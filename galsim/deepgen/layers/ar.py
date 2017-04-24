# This module contains a Lasagne implementation of the MADE
# model proposed by M. Germain (see https://github.com/mgermain/MADE)
# Most of this code is directly extracted from M. Germain implementation
import copy
import theano
import theano.tensor as T
import numpy as np

from theano.sandbox.rng_mrg import MRG_RandomStreams  # Limited but works on GPU
from theano.tensor.shared_randomstreams import RandomStreams

import lasagne
import lasagne.layers
import lasagne.init as init
import lasagne.nonlinearities as nonlinearities
from lasagne.layers import get_output_shape
from lasagne.layers import SliceLayer, batch_norm, ElemwiseSumLayer, NonlinearityLayer
from lasagne.layers import DenseLayer
from lasagne.layers.conv import BaseConvLayer
from lasagne.utils import floatX

class MaskGenerator(object):

    def __init__(self, input_size, hidden_sizes, l, random_seed=1234):
        self._random_seed = random_seed
        self._mrng = MRG_RandomStreams(seed=random_seed)
        self._rng = RandomStreams(seed=random_seed)

        self._hidden_sizes = hidden_sizes
        self._input_size = input_size
        self._l = l

        self.ordering = theano.shared(value=np.arange(input_size, dtype=theano.config.floatX), name='ordering', borrow=False)

        # Initial layer connectivity
        self.layers_connectivity = [theano.shared(value=(self.ordering + 1).eval(), name='layer_connectivity_input', borrow=False)]
        for i in range(len(self._hidden_sizes)):
            self.layers_connectivity += [theano.shared(value=np.zeros((self._hidden_sizes[i]), dtype=theano.config.floatX), name='layer_connectivity_hidden{0}'.format(i), borrow=False)]
        self.layers_connectivity += [self.ordering]

        ## Theano functions
        new_ordering = self._rng.shuffle_row_elements(self.ordering)
        self.shuffle_ordering = theano.function(name='shuffle_ordering',
                                                inputs=[],
                                                updates=[(self.ordering, new_ordering), (self.layers_connectivity[0], new_ordering + 1)])

        self.layers_connectivity_updates = []
        for i in range(len(self._hidden_sizes)):
            self.layers_connectivity_updates += [self._get_hidden_layer_connectivity(i)]
        # self.layers_connectivity_updates = [self._get_hidden_layer_connectivity(i) for i in range(len(self._hidden_sizes))]  # WTF THIS DO NOT WORK
        self.sample_connectivity = theano.function(name='sample_connectivity',
                                                   inputs=[],
                                                   updates=[(self.layers_connectivity[i+1], self.layers_connectivity_updates[i]) for i in range(len(self._hidden_sizes))])

        # Save random initial state
        self._initial_mrng_rstate = copy.deepcopy(self._mrng.rstate)
        self._initial_mrng_state_updates = [state_update[0].get_value() for state_update in self._mrng.state_updates]

        # Ensuring valid initial connectivity
        self.sample_connectivity()

    def reset(self):
        # Set Original ordering
        self.ordering.set_value(np.arange(self._input_size, dtype=theano.config.floatX))

        # Reset RandomStreams
        self._rng.seed(self._random_seed)

        # Initial layer connectivity
        self.layers_connectivity[0].set_value((self.ordering + 1).eval())
        for i in range(1, len(self.layers_connectivity)-1):
            self.layers_connectivity[i].set_value(np.zeros((self._hidden_sizes[i-1]), dtype=theano.config.floatX))
        self.layers_connectivity[-1].set_value(self.ordering.get_value())

        # Reset MRG_RandomStreams (GPU)
        self._mrng.rstate = self._initial_mrng_rstate
        for state, value in zip(self._mrng.state_updates, self._initial_mrng_state_updates):
            state[0].set_value(value)

        self.sample_connectivity()

    def _get_p(self, start_choice):
        start_choice_idx = (start_choice-1).astype('int32')
        p_vals = T.concatenate([T.zeros((start_choice_idx,)), T.nnet.nnet.softmax(self._l * T.arange(start_choice, self._input_size, dtype=theano.config.floatX))[0]])
        p_vals = T.inc_subtensor(p_vals[start_choice_idx], 1.)  # Stupid hack because de multinomial does not contain a safety for numerical imprecision.
        return p_vals

    def _get_hidden_layer_connectivity(self, layerIdx):
        layer_size = self._hidden_sizes[layerIdx]
        if layerIdx == 0:
            p_vals = self._get_p(T.min(self.layers_connectivity[layerIdx]))
        else:
            p_vals = self._get_p(T.min(self.layers_connectivity_updates[layerIdx-1]))

        # #Implementations of np.choose in theano GPU
        # return T.nonzero(self._mrng.multinomial(pvals=[self._p_vals] * layer_size, dtype=theano.config.floatX))[1].astype(dtype=theano.config.floatX)
        # return T.argmax(self._mrng.multinomial(pvals=[self._p_vals] * layer_size, dtype=theano.config.floatX), axis=1)
        return T.sum(T.cumsum(self._mrng.multinomial(pvals=T.tile(p_vals[::-1][None, :], (layer_size, 1)), dtype=theano.config.floatX), axis=1), axis=1)

    def _get_mask(self, layerIdxIn, layerIdxOut):
        return (self.layers_connectivity[layerIdxIn][:, None] <= self.layers_connectivity[layerIdxOut][None, :]).astype(theano.config.floatX)

    def get_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(layerIdx, layerIdx + 1)

    def get_direct_input_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(0, layerIdx)

    def get_direct_output_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(layerIdx, -1)


class MaskedLayer(DenseLayer):
    """
    This class does the both the masked layer and the conditioned masked layer
    """

    def __init__(self, incoming, num_units, layerIdx, mask_generator,
        use_cond_mask=False, **kwargs):
        super(MaskedLayer, self).__init__(incoming, num_units, **kwargs)
        self.layerIdx = layerIdx
        self.mask_generator = mask_generator
        self.use_cond_mask = use_cond_mask
        self.num_inputs = int(np.prod(self.input_shape[self.num_leading_axes:]))

        self.weights_mask = self.add_param(lasagne.init.Constant(1.),
                                           shape = (self.num_inputs, self.num_units),
                                           name='weights_mask',
                                           trainable=False,
                                           regularizable=False)

        if self.use_cond_mask:
            self.U = self.add_param(lasagne.init.GlorotUniform(),
                                               shape = (self.num_inputs, self.num_units),
                                               name='U')

        self.shuffle_update = [(self.weights_mask, mask_generator.get_mask_layer_UPDATE(self.layerIdx))]

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        activation = T.dot(input, self.W * self.weights_mask)
        if self.use_cond_mask:
            activation = activation + T.dot(T.ones_like(input), self.U * self.weights_mask)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)


class ARConv2DLayer(BaseConvLayer):
    """
    Layer implementing a masked autoregressive convolution
    """
    
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, W=init.GlorotUniform(), b=init.Constant(0.), untie_biases=False,
                 nonlinearity=nonlinearities.rectify, flip_filters=True, zerodiagonal=True,  flipmask=False,
                 convolution=T.nnet.conv2d, **kwargs):
        super(ARConv2DLayer, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad,  W=W,
                                          nonlinearity=nonlinearity, flip_filters=flip_filters, n=2, b=b, untie_biases=untie_biases,
                                          **kwargs)
        self.convolution = convolution
        
        w_shape = self.get_W_shape()
        
        # Create the convolution mask
        self.weights_mask = self.add_param(lasagne.init.Constant(1.),
                                           shape = w_shape,
                                           name='weights_mask',
                                           trainable=False,
                                           regularizable=False)
        # Initialise the mask
        # Adapted from https://github.com/openai/iaf/blob/master/graphy/nodes/ar.py
        l = w_shape[2] // 2 
        m = w_shape[3] // 2 
        self.weights_mask = T.set_subtensor(self.weights_mask[:,:,:l,:], floatX(0.0))
        self.weights_mask = T.set_subtensor(self.weights_mask[:,:,l,:m], floatX(0.0))
        n_out = w_shape[0]
        n_in = w_shape[1]
        
        if n_out >= n_in:
            assert n_out%n_in == 0
            k = n_out / n_in
            for i in range(n_in):
                self.weights_mask = T.set_subtensor(self.weights_mask[i*k:(i+1)*k,i+1:,l,m], floatX(0.0))
                if zerodiagonal:
                   self.weights_mask = T.set_subtensor(self.weights_mask[i*k:(i+1)*k,i:i+1,l,m],  floatX(0.0))
        else:
            assert n_in%n_out == 0
            k = n_in / n_out
            for i in range(n_out):
                self.weights_mask = T.set_subtensor(self.weights_mask[i:i+1,(i+1)*k:,l,m], floatX(0.0))
                if zerodiagonal:
                   self.weights_mask = T.set_subtensor(self.weights_mask[i:i+1,i*k:(i+1)*k:,l,m], floatX(0.0))
        if flipmask:
           self.weights_mask = self.weights_mask[::-1,::-1,::-1,::-1]
            
        # TODO: Renormalise the initial weights after masking...


    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, self.W * self.weights_mask,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved
    
