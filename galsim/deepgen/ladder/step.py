import theano.tensor as T
import theano

import lasagne
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity, softmax
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import TransposedConv2DLayer, Upscale2DLayer, Conv2DLayer, BatchNormLayer
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, ReshapeLayer, FlattenLayer, NonlinearityLayer, get_output, get_all_params, batch_norm, get_output_shape

from lasagne.init import HeNormal
from lasagne.layers import BatchNormLayer, NonlinearityLayer, SliceLayer, ElemwiseSumLayer, DropoutLayer

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

    
    def connect_upward(self, d, y):
        
        return d

    def connect_downward(self, p, y):
        
        return p

    def kl_normal(self, inputs):
        """
        Computes the kl divergence of the lattent variables in this layer
        """
        if self.qz_smpl is not None:
            # Computes the inference network posterior
            qz_mu, qz_logvar, pz_mu, pz_logvar = get_output([
                self.qz_mu, self.qz_logvar,
                self.pz_mu, self.pz_logvar], inputs=inputs)

            shape = get_output_shape(self.qz_mu)
            # Sum over output dimensions but not the batchsize
            return  kl_normal2_normal2(qz_mu, qz_logvar, pz_mu, pz_logvar, eps=1e-6).sum(axis=range(1,len(shape)))
        else:
            return 0

    def log_likelihood(self, inputs):
        """
        Computes the log likelihood of the model
        """
        x, x_mu = get_output([self.d_in, self.top_down_net],
                                         inputs=inputs)
        shape = get_output_shape(self.d_in)
        c = - 0.5 * math.log(2*math.pi)
        loglik = c - math.log(self.noise_std) - (x - x_mu)**2 / (2 * self.noise_std**2)
        return loglik.sum(axis=range(1,len(shape)))

class resnet_step(ladder_step):

    def __init__(self, n_filters=32, latent_dim=8, downsample=False, prefilter=False):
        """
        Initialise the step
        """
        self.noise_std=0.01
        self.n_filters = n_filters
        self.latent_dim = latent_dim
        self.prefilter = prefilter
        self.latent_dim = latent_dim
        self.downsample = downsample

    def connect_upward(self, d, y):
        he_norm = HeNormal(gain='relu')
        self.d_in = d
        
        # Get the dimension of the input and check if downsampling is requested 
        self.n_filters_in =  d.output_shape[1]    
        self.n_x =  d.output_shape[-1]    
        stride = 2 if self.downsample else 1

        # If the input needs to be reshaped in any way we do it first
        if self.prefilter:
            input_net = batch_norm(Conv2DLayer(d, num_filters=self.n_filters, filter_size=5, stride=stride, nonlinearity=elu, pad='same', W=he_norm))
            branch = input_net
        elif self.n_filters_in != self.n_filters or self.downsample:
            input_net = NonlinearityLayer(BatchNormLayer(d), elu)
            branch = input_net
        else:
            input_net = d
            branch = BatchNormLayer(d)
            branch = NonlinearityLayer(branch, elu)

        #
        # Main branch
        #
        branch = Conv2DLayer(branch, num_filters=self.n_filters, filter_size=3, stride=1, nonlinearity=identity, pad='same', W=he_norm)
        
        if self.latent_dim > 0:
            branch_posterior = SliceLayer(branch, indices=slice(-self.latent_dim, None), axis=1)
        
        branch = NonlinearityLayer(BatchNormLayer(branch), elu)
        branch = Conv2DLayer(branch, num_filters=self.n_filters, filter_size=3, stride=stride, nonlinearity=identity, pad='same', W=he_norm)

        #
        # Shortcut branch
        #
        if self.n_filters_in != self.n_filters or self.downsample:
            shortcut = Conv2DLayer(input_net, num_filters=self.n_filters, filter_size=1, nonlinearity=None, pad='same', stride=stride, W=he_norm)
        else:
            shortcut = input_net

        self.bottom_up_net = ElemwiseSumLayer([branch, shortcut])
        
        if self.latent_dim > 0:
            # Encoding the posterior
            self.d_mu = Conv2DLayer(branch_posterior, num_filters=self.latent_dim,
                                    filter_size=1, nonlinearity=None, pad='same',
                                    W=GlorotUniform())
            self.d_logvar = ClampLogVarLayer(Conv2DLayer(branch_posterior,
                                                        num_filters=self.latent_dim,
                                                        filter_size=1, nonlinearity=None, pad='same', W=GlorotUniform()))
            self.d_smpl = GaussianSampleLayer(mean=self.d_mu, log_var=self.d_logvar)

        return self.bottom_up_net


    def connect_downward(self, p, y):
        he_norm = HeNormal(gain='relu')
        
        # Get the dimension of the input and check if downsampling is requested 
        stride = 2 if self.downsample else 1

        # If the input needs to be reshaped in any way we do it first
        if self.n_filters_in != self.n_filters or self.downsample:
            input_net = NonlinearityLayer(BatchNormLayer(p), elu)
            branch = input_net
        else:
            input_net = p
            branch = BatchNormLayer(p)
            branch = NonlinearityLayer(branch, elu)

        if self.latent_dim > 0:
            #
            # Inference branch
            #
            if self.downsample :
                branch_posterior = TransposedConv2DLayer(branch,
                                                        num_filters=self.latent_dim,
                                                        filter_size=3, 
                                                        stride=stride, nonlinearity=identity,
                                                        crop='same',
                                                        output_size=self.n_x,W=he_norm)
            else:
                branch_posterior = Conv2DLayer(branch,
                                            num_filters=self.latent_dim,
                                            filter_size=3, 
                                            stride=stride,
                                            nonlinearity=identity,
                                            pad='same',
                                            W=he_norm)
                
            tz_mu = Conv2DLayer(branch_posterior, num_filters=self.latent_dim, filter_size=1, stride=1,pad='same', nonlinearity=None, W=he_norm)
            tz_logvar = ClampLogVarLayer(Conv2DLayer(branch_posterior, num_filters=self.latent_dim, filter_size=1, stride=1,pad='same', nonlinearity=None, W=he_norm))

            # Combine top-down inference information
            self.qz_mu = MergeMeanLayer(self.d_mu, self.d_logvar, tz_mu, tz_logvar)
            self.qz_logvar = MergeLogVarLayer(self.d_logvar, tz_logvar)
            self.qz_smpl = GaussianSampleLayer(mean=self.qz_mu, log_var=self.qz_logvar)
        else:
            self.qz_smpl = None    


        #
        # Main branch
        # 
        if self.downsample :
            branch = TransposedConv2DLayer(branch,
                                            num_filters=self.n_filters,
                                            filter_size=3,
                                            stride=stride, nonlinearity=identity,
                                            crop='same',
                                            output_size=self.n_x,W=he_norm)
        else:
            branch = Conv2DLayer(branch,
                                num_filters=self.n_filters,
                                filter_size=3, 
                                stride=stride,
                                nonlinearity=identity,
                                pad='same',
                                W=he_norm)
        if self.latent_dim > 0:
            print(get_output_shape(branch))
            branch_prior = SliceLayer(branch, indices=slice(-self.latent_dim, None), axis=1)
            branch = SliceLayer(branch, indices=slice(0,-self.latent_dim), axis=1)
            print(get_output_shape(branch_prior))
            print(get_output_shape(branch))
            ## Merge samples from the posterior into the main branch
            branch = ConcatLayer([branch, self.qz_smpl])
            print(get_output_shape(branch))

        branch = NonlinearityLayer(BatchNormLayer(branch), elu)
        branch = Conv2DLayer(branch,
                             num_filters=self.n_filters_in,
                             filter_size=3, 
                             stride=1,
                             nonlinearity=identity,
                             pad='same',
                             W=he_norm)

        if self.latent_dim > 0:
            # Define step prior
            self.pz_mu = Conv2DLayer(branch_prior, num_filters=self.latent_dim, filter_size=1, stride=1,pad='same', nonlinearity=None, W=he_norm)
            self.pz_logvar = ClampLogVarLayer(Conv2DLayer(branch_prior, num_filters=self.latent_dim, filter_size=1, stride=1,pad='same', nonlinearity=None, W=he_norm))
            self.pz_smpl = GaussianSampleLayer(mean=self.pz_mu, log_var=self.pz_logvar)
        else:
            self.pz_smpl = None
            
        if self.n_filters_in != self.n_filters or self.downsample:
            shortcut = TransposedConv2DLayer(input_net, num_filters=self.n_filters_in, filter_size=1, nonlinearity=None,  stride=stride, W=he_norm, crop='same',
            output_size=self.n_x)
        else:
            shortcut = input_net
        
        net = ElemwiseSumLayer([branch, shortcut])
        
        if self.prefilter:
            net = Conv2DLayer(net, num_filters=self.n_filters_in, filter_size=5, stride=1, nonlinearity=identity, pad='same', W=he_norm)
            
        self.top_down_net = net
        return self.top_down_net


class dens_step(ladder_step):

    def __init__(self, n_units=128, nonlinearity=elu):
        """

        """
        self.n_units = n_units
        self.nonlinearity = nonlinearity
    
    def connect_upward(self, d, y):
        he_norm = HeNormal(gain='relu')
        
        # Saves the input layer shape for latter
        self.input_shape = get_output_shape(d)
        self.d_in = d
        
        # Concatenate the two input layers
        input_net  = FlattenLayer(d)
        self.n_units_in = get_output_shape(input_net)[-1]
        
        branch = ConcatLayer([input_net, y])

        #
        # Main branch
        #
        branch = batch_norm(DenseLayer(branch, num_units=self.n_units_in,
                                    nonlinearity=elu,
                                    W=he_norm))

        branch = batch_norm(DenseLayer(branch, num_units=self.n_units,
                                    nonlinearity=elu,
                                    W=he_norm))
        

        self.bottom_up_net = branch
        

        return self.bottom_up_net


    def connect_downward(self, p, y):
        he_norm = HeNormal(gain='relu')

        # If the input needs to be reshaped in any way we do it first

        branch = ConcatLayer([FlattenLayer(p), y])
            

        #
        # Main branch
        # 
        branch = batch_norm(DenseLayer(branch, num_units=self.n_units, nonlinearity=elu, W=he_norm))
        
        branch = batch_norm(DenseLayer(branch, num_units=self.n_units_in, nonlinearity=elu, W=he_norm))
        self.qz_smpl = None
        self.pz_smpl = None
        
        self.top_down_net = ReshapeLayer(branch,  shape=self.input_shape)
        return self.top_down_net


    def kl_normal(self, inputs):
        return 0


class dens_res_step(ladder_step):

    def __init__(self, latent_dim=16, n_units=128, nonlinearity=elu):
        """

        """
        self.n_units = n_units
        self.latent_dim = latent_dim
        self.nonlinearity = nonlinearity
    
    def connect_upward(self, d, y):
        he_norm = HeNormal(gain='relu')
        
        # Saves the input layer shape for latter
        self.input_shape = get_output_shape(d)
        self.d_in = d
        
        # Concatenate the two input layers
        input_net  = FlattenLayer(d)
        self.n_units_in = get_output_shape(input_net)[-1]
        
        if self.n_units != self.n_units_in:
            input_net = NonlinearityLayer(BatchNormLayer(input_net), elu)
            branch = ConcatLayer([input_net, y])
        else:
            branch = NonlinearityLayer(BatchNormLayer(input_net), elu)
            branch = ConcatLayer([branch, y])

        #
        # Main branch
        #
        branch = DenseLayer(branch, num_units=self.n_units_in,
                                    nonlinearity=identity,
                                    W=he_norm)

        branch_posterior = SliceLayer(branch, indices=slice(-self.latent_dim, None), axis=1)
        
        branch = NonlinearityLayer(BatchNormLayer(branch), elu)
        branch = DenseLayer(branch, num_units=self.n_units,
                                    nonlinearity=identity,
                                    W=he_norm)

        #
        # Shortcut branch
        #
        if self.n_units != self.n_units_in:
            shortcut = DenseLayer(branch, num_units=self.n_units,
                                    nonlinearity=identity,
                                    W=he_norm)
        else:
            shortcut = input_net

        self.bottom_up_net = ElemwiseSumLayer([branch, shortcut])
        
        # Encoding the posterior
        self.d_mu = DenseLayer(branch_posterior, num_units=self.latent_dim,nonlinearity=None, W=GlorotUniform())
        self.d_logvar = ClampLogVarLayer(DenseLayer(branch_posterior, num_units=self.latent_dim,nonlinearity=None, W=GlorotUniform()))
        self.d_smpl = GaussianSampleLayer(mean=self.d_mu, log_var=self.d_logvar)

        return self.bottom_up_net


    def connect_downward(self, p, y):
        he_norm = HeNormal(gain='relu')

        # If the input needs to be reshaped in any way we do it first
        if self.n_units != self.n_units_in:
            input_net = NonlinearityLayer(BatchNormLayer(p), elu)
            branch = ConcatLayer([FlattenLayer(input_net), y])
        else:
            input_net = p
            branch =  NonlinearityLayer(BatchNormLayer(p), elu)
            branch =  ConcatLayer([FlattenLayer(branch), y])


        #
        # Inference branch
        #
        branch_posterior = DenseLayer(branch, num_units=self.latent_dim, nonlinearity=identity, W=he_norm)
        tz_mu = DenseLayer(branch_posterior, num_units=self.latent_dim, nonlinearity=None, W=he_norm)
        tz_logvar = ClampLogVarLayer(DenseLayer(branch_posterior, num_units=self.latent_dim, nonlinearity=None, W=he_norm))

        # Combine top-down inference information
        self.qz_mu = MergeMeanLayer(self.d_mu, self.d_logvar, tz_mu, tz_logvar)
        self.qz_logvar = MergeLogVarLayer(self.d_logvar, tz_logvar)
        self.qz_smpl = GaussianSampleLayer(mean=self.qz_mu, log_var=self.qz_logvar)


        #
        # Main branch
        # 
        branch = DenseLayer(branch, num_units=self.n_units_in, nonlinearity=identity, W=he_norm)
        print(get_output_shape(branch))
        branch_prior = SliceLayer(branch, indices=slice(-self.latent_dim, None), axis=1)
        branch = SliceLayer(branch, indices=slice(0,-self.latent_dim), axis=1)
        print(get_output_shape(branch))
        print(get_output_shape(branch_prior))
        # Merge samples from the posterior into the main branch
        branch = ConcatLayer([branch, self.qz_smpl])
        print(get_output_shape(branch))

        branch = NonlinearityLayer(BatchNormLayer(branch), elu)
        branch = DenseLayer(branch, num_units=self.n_units_in, nonlinearity=identity, W=he_norm)

        # Define step prior
        self.pz_mu = DenseLayer(branch_prior, num_units=self.latent_dim, nonlinearity=identity, W=he_norm)
        self.pz_logvar = ClampLogVarLayer(DenseLayer(branch_prior, num_units=self.latent_dim, nonlinearity=identity, W=he_norm))
        self.pz_smpl = GaussianSampleLayer(mean=self.pz_mu, log_var=self.pz_logvar)
        
        if self.n_units != self.n_units_in:
            shortcut = DenseLayer(input_net, num_units=self.n_units_in, nonlinearity=None, W=he_norm)
        else:
            shortcut = input_net
        
        net = ElemwiseSumLayer([branch, shortcut])
        
        self.top_down_net = ReshapeLayer(net,  shape=self.input_shape)
        return self.top_down_net

