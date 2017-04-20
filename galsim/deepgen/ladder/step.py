from abc import ABCMeta

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
from ..blocks.IAF import ConvAR_IAF
from ..distributions import kl_normal2_normal2, log_normal2, log_bernoulli

import math

class ladder_step():
    
    __metaclass__ = ABCMeta
    """
    Base element of a Ladder
    """

    @abstractmethod
    def connect_upward(self, d, y):
        """
        Connects the bottom-up part of the ladder
        """
        self.d_in = d


    @abstractmethod
    def connect_downward(self, p, y):
        """
        Connects the top-down part of the ladder
        """
        self.log_pz = None
        self.log_qz = None
        self.top_down_net = p 

    def kl(self, inputs):
        """
        Computes the kl divergence of the lattent variables in this layer
        """
        if self.log_pz is not None and self.log_qz is not None:
            pz, qz = get_output([self.log_pz, self.log_qz], inputs=inputs)
            return pz - qz
        else:
            return 0

    def gaussian_likelihood(self, inputs, noise_std=1):
        """
        Computes the log likelihood assuming a Gaussian model
        """
        x, x_mu = get_output([self.d_in, self.top_down_net], inputs=inputs)
        shape = get_output_shape(self.d_in)
        c = - 0.5 * math.log(2*math.pi)
        log_px_given_z = c - math.log(noise_std) - (x - x_mu)**2 / (2 * noise_std**2)
        return log_px_given_z.sum(axis=range(1,len(shape)))
    
    
    def bernoulli_likelihood(self, inputs):
        """
        Computes the log likelihood assuming a Bernoulli model
        """
        x, x_mu = get_output([self.d_in, self.top_down_net], inputs=inputs)
        shape = get_output_shape(self.d_in)
        log_px_given_z = log_bernoulli(x, x_mu, eps=1e-7)
        return log_px_given_z.sum(axis=range(1,len(shape)))


class resnet_step(ladder_step):

    def __init__(self, n_filters=32, latent_dim=0, IAF_sizes=[], downsample=False, prefilter=False, output_nonlinearity=identity):
        """
        Initialise the step
        """
        self.n_filters = n_filters
        self.latent_dim = latent_dim
        self.prefilter = prefilter
        self.latent_dim = latent_dim
        self.downsample = downsample
        self.IAF_sizes = IAF_sizes
        self.output_nonlinearity = output_nonlinearity


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
            branch_posterior = SliceLayer(branch, indices=slice(-2*self.latent_dim, None), axis=1)
        
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
            # Encoding for the posterior
            self.d_mu = SliceLayer(branch_posterior, indices=slice(0,self.latent_dim), axis=1)
            self.d_logvar = ClampLogVarLayer(SliceLayer(branch_posterior, indices=slice(-self.latent_dim, None), axis=1))

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
                                                        num_filters=2*self.latent_dim,
                                                        filter_size=3, 
                                                        stride=stride, nonlinearity=identity,
                                                        crop='same',
                                                        output_size=self.n_x,W=he_norm)
            else:
                branch_posterior = Conv2DLayer(branch,
                                            num_filters=2*self.latent_dim,
                                            filter_size=3, 
                                            stride=stride,
                                            nonlinearity=identity,
                                            pad='same',
                                            W=he_norm)
            tz_mu = SliceLayer(branch_posterior, indices=slice(0,self.latent_dim), axis=1)
            tz_logvar = ClampLogVarLayer(SliceLayer(branch_posterior, indices=slice(-self.latent_dim, None), axis=1))

            # Combine top-down inference information
            self.qz_mu = MergeMeanLayer(self.d_mu, self.d_logvar, tz_mu, tz_logvar)
            self.qz_logvar = MergeLogVarLayer(self.d_logvar, tz_logvar)
            
            # Sample at the origin of the IAF chain
            self.qz0 = GaussianSampleLayer(mean=self.qz_mu, log_var=self.qz_logvar)
            
            # Add the Autoregressive layers
            z = self.qz0
            for i,h_size in enumerate(self.IAF_sizes):
                flipmask = ((i % 2) == 1)
                m = z
                for j in h_size:
                    m = ARConv2DLayer(m, num_filters=j, filter_size=3, pad='same', nonlinearity=elu, flipmask=flipmask)
                m = ARConv2DLayer(m, num_filters=latent_dim, filter_size=3, pad='same', nonlinearity=identity, flipmask=flipmask)
                
                # Only parametrising the mean, following recommendations from Kingma
                z = ElemwiseSumLayer([z, m])
                
            self.qz_smpl = z
            self.log_qz = LogNormalLayer([self.qz0, self.qz_mu, self.qz_logvar], epsilon=1e-7)
        else:  
            self.log_qz = None


        #
        # Main branch
        # 
        if self.downsample :
            branch = TransposedConv2DLayer(branch,
                                            num_filters=self.n_filters+self.latent_dim,
                                            filter_size=3,
                                            stride=stride, nonlinearity=identity,
                                            crop='same',
                                            output_size=self.n_x,W=he_norm)
        else:
            branch = Conv2DLayer(branch,
                                num_filters=self.n_filters+self.latent_dim,
                                filter_size=3, 
                                stride=stride,
                                nonlinearity=identity,
                                pad='same',
                                W=he_norm)
        if self.latent_dim > 0:
            branch_prior = SliceLayer(branch, indices=slice(-2*self.latent_dim, None), axis=1)
            branch = SliceLayer(branch, indices=slice(0,-2*self.latent_dim), axis=1)
            ## Merge samples from the posterior into the main branch
            branch = ConcatLayer([branch, self.qz_smpl])

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
            self.pz_mu = SliceLayer(branch_prior, indices=slice(0,self.latent_dim), axis=1)
            self.pz_logvar = ClampLogVarLayer(SliceLayer(branch_prior, indices=slice(-self.latent_dim, None), axis=1))
            self.pz_smpl = GaussianSampleLayer(mean=self.pz_mu, log_var=self.pz_logvar)
            self.log_pz = LogNormalLayer([self.qz_smpl, self.pz_mu, self.pz_logvar], epsilon=1e-7)
        else:
            self.log_pz = None
            
        if self.n_filters_in != self.n_filters or self.downsample:
            shortcut = TransposedConv2DLayer(input_net, num_filters=self.n_filters_in, filter_size=1, nonlinearity=None,  stride=stride, W=he_norm, crop='same',
            output_size=self.n_x)
        else:
            shortcut = input_net
        
        net = ElemwiseSumLayer([branch, shortcut])
        
        if self.prefilter:
            net = Conv2DLayer(net, num_filters=self.n_filters_in, filter_size=5, stride=1, nonlinearity=self.output_nonlinearity, pad='same', W=he_norm)

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

        self.qz_smpl = None
        self.pz_smpl = None
        self.log_qz = None
        self.log_pz = None
        
        # If the input needs to be reshaped in any way we do it first

        branch = ConcatLayer([FlattenLayer(p), y])

        #
        # Main branch
        # 
        branch = batch_norm(DenseLayer(branch, num_units=self.n_units, nonlinearity=elu, W=he_norm))
        
        branch = batch_norm(DenseLayer(branch, num_units=self.n_units_in, nonlinearity=elu, W=he_norm))
        
        self.top_down_net = ReshapeLayer(branch,  shape=self.input_shape)
        return self.top_down_net


#class dens_res_step(ladder_step):

    #def __init__(self, latent_dim=16, n_units=128, nonlinearity=elu):
        #"""

        #"""
        #self.n_units = n_units
        #self.latent_dim = latent_dim
        #self.nonlinearity = nonlinearity
    
    #def connect_upward(self, d, y):
        #he_norm = HeNormal(gain='relu')
        
        ## Saves the input layer shape for latter
        #self.input_shape = get_output_shape(d)
        #self.d_in = d
        
        ## Concatenate the two input layers
        #input_net  = FlattenLayer(d)
        #self.n_units_in = get_output_shape(input_net)[-1]
        
        #if self.n_units != self.n_units_in:
            #input_net = NonlinearityLayer(BatchNormLayer(input_net), elu)
            #branch = ConcatLayer([input_net, y])
        #else:
            #branch = NonlinearityLayer(BatchNormLayer(input_net), elu)
            #branch = ConcatLayer([branch, y])

        ##
        ## Main branch
        ##
        #branch = DenseLayer(branch, num_units=self.n_units_in,
                                    #nonlinearity=identity,
                                    #W=he_norm)

        #branch_posterior = SliceLayer(branch, indices=slice(-self.latent_dim, None), axis=1)
        
        #branch = NonlinearityLayer(BatchNormLayer(branch), elu)
        #branch = DenseLayer(branch, num_units=self.n_units,
                                    #nonlinearity=identity,
                                    #W=he_norm)

        ##
        ## Shortcut branch
        ##
        #if self.n_units != self.n_units_in:
            #shortcut = DenseLayer(branch, num_units=self.n_units,
                                    #nonlinearity=identity,
                                    #W=he_norm)
        #else:
            #shortcut = input_net

        #self.bottom_up_net = ElemwiseSumLayer([branch, shortcut])
        
        ## Encoding the posterior
        #self.d_mu = DenseLayer(branch_posterior, num_units=self.latent_dim,nonlinearity=None, W=GlorotUniform())
        #self.d_logvar = ClampLogVarLayer(DenseLayer(branch_posterior, num_units=self.latent_dim,nonlinearity=None, W=GlorotUniform()))
        #self.d_smpl = GaussianSampleLayer(mean=self.d_mu, log_var=self.d_logvar)

        #return self.bottom_up_net


    #def connect_downward(self, p, y):
        #he_norm = HeNormal(gain='relu')

        ## If the input needs to be reshaped in any way we do it first
        #if self.n_units != self.n_units_in:
            #input_net = NonlinearityLayer(BatchNormLayer(p), elu)
            #branch = ConcatLayer([FlattenLayer(input_net), y])
        #else:
            #input_net = p
            #branch =  NonlinearityLayer(BatchNormLayer(p), elu)
            #branch =  ConcatLayer([FlattenLayer(branch), y])


        ##
        ## Inference branch
        ##
        #branch_posterior = DenseLayer(branch, num_units=self.latent_dim, nonlinearity=identity, W=he_norm)
        #tz_mu = DenseLayer(branch_posterior, num_units=self.latent_dim, nonlinearity=None, W=he_norm)
        #tz_logvar = ClampLogVarLayer(DenseLayer(branch_posterior, num_units=self.latent_dim, nonlinearity=None, W=he_norm))

        ## Combine top-down inference information
        #self.qz_mu = MergeMeanLayer(self.d_mu, self.d_logvar, tz_mu, tz_logvar)
        #self.qz_logvar = MergeLogVarLayer(self.d_logvar, tz_logvar)
        #self.qz_smpl = GaussianSampleLayer(mean=self.qz_mu, log_var=self.qz_logvar)


        ##
        ## Main branch
        ## 
        #branch = DenseLayer(branch, num_units=self.n_units_in, nonlinearity=identity, W=he_norm)
        #print(get_output_shape(branch))
        #branch_prior = SliceLayer(branch, indices=slice(-self.latent_dim, None), axis=1)
        #branch = SliceLayer(branch, indices=slice(0,-self.latent_dim), axis=1)
        #print(get_output_shape(branch))
        #print(get_output_shape(branch_prior))
        ## Merge samples from the posterior into the main branch
        #branch = ConcatLayer([branch, self.qz_smpl])
        #print(get_output_shape(branch))

        #branch = NonlinearityLayer(BatchNormLayer(branch), elu)
        #branch = DenseLayer(branch, num_units=self.n_units_in, nonlinearity=identity, W=he_norm)

        ## Define step prior
        #self.pz_mu = DenseLayer(branch_prior, num_units=self.latent_dim, nonlinearity=identity, W=he_norm)
        #self.pz_logvar = ClampLogVarLayer(DenseLayer(branch_prior, num_units=self.latent_dim, nonlinearity=identity, W=he_norm))
        #self.pz_smpl = GaussianSampleLayer(mean=self.pz_mu, log_var=self.pz_logvar)
        
        #if self.n_units != self.n_units_in:
            #shortcut = DenseLayer(input_net, num_units=self.n_units_in, nonlinearity=None, W=he_norm)
        #else:
            #shortcut = input_net
        
        #net = ElemwiseSumLayer([branch, shortcut])
        
        #self.top_down_net = ReshapeLayer(net,  shape=self.input_shape)
        #return self.top_down_net

