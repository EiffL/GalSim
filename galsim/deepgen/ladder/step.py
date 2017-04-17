import theano.tensor as T
import theano

import lasagne
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity, softmax
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import TransposedConv2DLayer, Upscale2DLayer, Conv2DLayer, BatchNormLayer
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, ReshapeLayer, FlattenLayer, NonlinearityLayer, get_output, get_all_params, batch_norm, get_output_shape

from ..layers.sample import GaussianSampleLayer, BernoulliSampleLayer
from ..layers.transform import ScaleShiftLayer, CropLayer, ClampLogVarLayer
from ..blocks.resnet import preactivation_resnet, transposed_preactivation_resnet
from ..distributions import kl_normal2_normal2, log_normal2, log_bernoulli

import math

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

class ladder_step():
    """
    Base element of a Ladder
    """

    def __init__(self, noise_std=0.01):
        """
        Defines the noise variance to use in the  log likelihood
        """
        self.noise_std = noise_std

    def bottom_up(self, input_layer, y):
        """
        Creates the bottom up architecture of the block
        """
        return input_layer

    def top_down(self, input_layer, y):
        """
        Creates the top down architecture of the block
        """
        return input_layer

    def connect_upward(self, d, y):
        self.d_in = d
        self.d_mu, self.d_logvar, self.d_smpl = self.bottom_up(d, y)
        return self.d_mu

    def connect_downward(self, p, y, qz_smpl=None, tz_mu=None, tz_logvar=None):
        assert (qz_smpl is not None) or ((tz_mu is not None) and (tz_logvar is not None))

        self.p_mu, self.p_logvar, self.p_smpl = self.top_down(p, y)

        if qz_smpl is None:
            # Combine top-down inference information
            self.qz_mu = MergeMeanLayer(self.d_mu, self.d_logvar, tz_mu, tz_logvar)
            self.qz_logvar = MergeLogVarLayer(self.d_logvar, tz_logvar)
            self.qz_smpl = GaussianSampleLayer(mean=self.qz_mu, log_var=self.qz_logvar)
        else:
            self.qz_smpl = qz_smpl

        self.t_mu, self.t_logvar, self.t_smpl = self.top_down(self.qz_smpl, y)
        return self.p_smpl, self.t_mu, self.t_logvar

    def kl_normal(self, inputs, top_step):
        """
        Computes the kl divergence between this layer and
        """
        # Computes the inference network posterior
        qz_mu, qz_logvar, pz_mu, pz_logvar = get_output([
            self.qz_mu, self.qz_logvar,
            top_step.p_mu, top_step.p_logvar], inputs=inputs)

        shape = get_output_shape(self.qz_mu)
        # Sum over output dimensions but not the batchsize
        return  kl_normal2_normal2(qz_mu, qz_logvar, pz_mu, pz_logvar, eps=1e-6).clip(0.125,100).sum(axis=range(1,len(shape)))

    def log_likelihood(self, inputs):
        """
        Computes the log likelihood of the model
        """
        x, x_mu, x_log_var = get_output([self.d_in, self.p_mu, self.p_logvar],
                                         inputs=inputs)
        shape = get_output_shape(self.d_in)
        c = - 0.5 * math.log(2*math.pi)
        loglik = c - math.log(self.noise_std) - (x - x_mu)**2 / (2 * self.noise_std**2)
        return loglik.sum(axis=range(1,len(shape)))

class resnet_step(ladder_step):

    def __init__(self, n_filters_in=3, n_filters=[32, 64, 128], latent_dim=256, resnet_per_stage=2, prefilter=True):
        """
        Initialise the step
        """
        self.noise_std=0.01
        self.n_filters = n_filters
        self.resnet_per_stage = resnet_per_stage
        self.latent_dim = latent_dim
        self.prefilter = prefilter
        self.n_filters_in = n_filters_in


    def bottom_up(self, input_layer, y):
        """
        Compute bottom up pass
        """
        input_dim = get_output_shape(input_layer)[1]

        if self.prefilter:
            network = Conv2DLayer(input_layer, num_filters=self.n_filters[0], filter_size=1, pad='same', stride=1, nonlinearity=None, W=GlorotUniform())
        else:
            network = input_layer

        for i in range(self.resnet_per_stage):
            network = preactivation_resnet(network, n_out_filters=self.n_filters[0], filter_size=3)

        for n_filter in self.n_filters[1:]:
            network = preactivation_resnet(network, n_out_filters=n_filter, filter_size=3, downsample=True)
            for i in range(self.resnet_per_stage):
                network = preactivation_resnet(network, n_out_filters=n_filter, filter_size=3)

        # Creating mu and sigma layers
        mu = Conv2DLayer(network, num_filters=self.latent_dim, filter_size=1, nonlinearity=None, pad='same', W=GlorotUniform())
        logvar = ClampLogVarLayer(Conv2DLayer(network, num_filters=self.latent_dim, filter_size=1, nonlinearity=None, pad='same', W=GlorotUniform()))
        samp = GaussianSampleLayer(mean=mu, log_var=logvar)

        return mu, logvar, samp

    def top_down(self, input_layer, y):
        """
        Compute top down pass
        """
        input_dim = get_output_shape(input_layer)[1]

        network = input_layer

        for i in range(self.resnet_per_stage):
            network = transposed_preactivation_resnet(network, n_out_filters=self.n_filters[-1], filter_size=3)

        for n_filter in self.n_filters[::-1][1:]:
            network = transposed_preactivation_resnet(network, n_out_filters=n_filter, filter_size=3, upsample=True)
            for i in range(self.resnet_per_stage):
                network = transposed_preactivation_resnet(network, n_out_filters=n_filter, filter_size=3)

        # Computing
        mu = Conv2DLayer(network, num_filters=self.n_filters_in, filter_size=1, stride=1,pad='same', nonlinearity=None, W=GlorotUniform())
        logvar = ClampLogVarLayer(Conv2DLayer(network, num_filters=self.n_filters_in, filter_size=1, stride=1,pad='same', nonlinearity=None, W=GlorotUniform()))
        smpl = GaussianSampleLayer(mean=mu, log_var=logvar)
        return mu, logvar, smpl

class dens_step(ladder_step):

    def __init__(self, n_out, n_units=[128, 128], nonlinearity=elu):
        """

        """
        self.output_shape = n_out
        self.n_units = n_units
        self.nonlinearity = nonlinearity

    def bottom_up(self, input_layer, y):
        """
        Copmute bottom up pass through a dense layer
        """
        
        # Saves the input layer shape for latter
        self.input_shape = get_output_shape(input_layer)
        
        # Concatenate the two input layers
        network = ConcatLayer([FlattenLayer(input_layer), y])

        for i in range(len(self.n_units)):
            network = batch_norm(DenseLayer(network, num_units=self.n_units[i],
                                          nonlinearity=self.nonlinearity,
                                          W=GlorotUniform(),
                                          name="rec_%d" % i))

        mu = DenseLayer(network, num_units=self.output_shape, nonlinearity=identity)
        logvar = ClampLogVarLayer(DenseLayer(network, num_units=self.output_shape, nonlinearity=identity))
        smpl = GaussianSampleLayer(mean=mu, log_var=logvar, name='z')

        return mu, logvar, smpl

    def top_down(self, input_layer, y):
        """
        Compute a top down pass through a dense layer
        """
        network = ConcatLayer([FlattenLayer(input_layer), y])

        for i in range(len(self.n_units)):
            network = batch_norm(DenseLayer(network, num_units=self.n_units[-i-1],
                                          nonlinearity=self.nonlinearity,
                                          W=GlorotUniform()))
        n_out = 1
        for i in range(len(self.input_shape)-1):
            n_out *= self.input_shape[i+1]

        mu = ReshapeLayer(DenseLayer(network, num_units=n_out, nonlinearity=identity), shape=self.input_shape)
        logvar = ClampLogVarLayer(ReshapeLayer(DenseLayer(network, num_units=n_out, nonlinearity=identity), shape=self.input_shape))

        smpl = GaussianSampleLayer(mean=mu, log_var=logvar, name='z')

        return mu, logvar, smpl
