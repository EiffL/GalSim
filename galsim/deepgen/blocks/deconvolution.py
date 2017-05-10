import theano
import theano.tensor as T

import lasagne
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity, softmax
from lasagne.layers import TransposedConv2DLayer, Upscale2DLayer, Conv2DLayer,get_output_shape

from ..layers.transform import SubpixelLayer

def deconvolution(net_input, num_filters, stride=2, filter_size=3, nonlinearity=identity, method='NN'):
    """
    Performs a deconvolution using one of the following methods
        - TransposeConvolution
        - Upsampling
    """
    if method == 'transposeConv':
        net = TransposedConv2DLayer(net_input, 
                                    num_filters=num_filters,
                                    filter_size=filter_size,
                                    stride=stride,
                                    nonlinearity=nonlinearity,
                                    crop='same',
                                    output_size=stride*get_output_shape(net_input)[-1])
    elif method == 'NN':
        net = Upscale2DLayer(net_input, stride, mode='repeat')
        net = Conv2DLayer(net, num_filters=num_filters, filter_size=filter_size,
                          nonlinearity=nonlinearity, pad='same')
    elif method == 'subpix':
        net = Conv2DLayer(net_input, num_filters=num_filters*stride**2, filter_size=filter_size,
                          nonlinearity=nonlinearity, pad='same')
        net = SubpixelLayer(net, stride, num_filters)
    else:
        print("Unknown deconvolution scheme")

    return net
