import lasagne
from lasagne.layers import BatchNormLayer, NonlinearityLayer, batch_norm, ElemwiseSumLayer, DropoutLayer
from lasagne.nonlinearities import elu
from lasagne.layers.dnn import Conv2DDNNLayer, Pool2DDNNLayer
from lasagne.init import HeNormal
from lasagne.layers import TransposedConv2DLayer

def preactivation_resnet(input_net, n_out_filters, filter_size=3, downsample=False):
    """
    Standard preactivation resnet, based on Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    """
    he_norm = HeNormal(gain='relu')
    n_filters =  input_net.output_shape[1]
    
    stride = 2 if downsample else 1

    # Main branch
    net = NonlinearityLayer(input_net, elu)    
    net = Conv2DDNNLayer(net, num_filters=n_out_filters, filter_size=filter_size, stride=1, nonlinearity=elu, pad='same', W=he_norm)
    net = batch_norm(Conv2DDNNLayer(net, num_filters=n_out_filters, filter_size=filter_size, stride=stride, nonlinearity=None, pad='same', W=he_norm))

    # Shortcut branch
    if n_filters != n_out_filters or downsample:
        shortcut = Conv2DDNNLayer(input_net, num_filters=n_out_filters, filter_size=filter_size, nonlinearity=None, pad='same', stride=stride, W=he_norm)
    else:
        shortcut = input_net

    return ElemwiseSumLayer([net, shortcut])


def transposed_preactivation_resnet(input_net, n_out_filters, filter_size=3, upsample=False):
    """
    Standard preactivation resnet, based on Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    """
    he_norm = HeNormal(gain='relu')
    n_filters =  input_net.output_shape[1]
    n_x = input_net.output_shape[-1]
    
    stride = 2 if upsample else 1
    
    # Main branch
    net = NonlinearityLayer(input_net, elu)
    if upsample:
        net = TransposedConv2DLayer(net, num_filters=n_out_filters, filter_size=filter_size, 
                                      stride=stride, nonlinearity=elu, crop='same', output_size=n_x*2)
    else:
        net = Conv2DDNNLayer(net, num_filters=n_out_filters, filter_size=filter_size, stride=1, nonlinearity=elu, pad='same', W=he_norm)
    net = batch_norm(Conv2DDNNLayer(net, num_filters=n_out_filters, filter_size=filter_size, stride=1, nonlinearity=None, W=he_norm, pad='same'))
    
    # Shortcut branch
    if upsample:
        shortcut = TransposedConv2DLayer(input_net, num_filters=n_out_filters, filter_size=filter_size, 
                                      stride=stride, nonlinearity=None, crop='same', output_size=n_x*2)
    elif n_filters != n_out_filters:
        shortcut = Conv2DDNNLayer(net, num_filters=n_out_filters, filter_size=filter_size, nonlinearity=None, pad='same', stride=stride, W=he_norm)
    else:
        shortcut = input_net
    
    return ElemwiseSumLayer([net, shortcut])


