from abc import ABCMeta, abstractmethod

import theano.tensor as T
import theano

import lasagne
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity, softmax
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import TransposedConv2DLayer, Upscale2DLayer, Conv2DLayer, BatchNormLayer
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, ReshapeLayer, FlattenLayer, NonlinearityLayer, get_output, get_all_params, batch_norm, get_output_shape

from lasagne.init import HeNormal
from lasagne.layers import BatchNormLayer, NonlinearityLayer, SliceLayer, ElemwiseSumLayer, DropoutLayer

from ..layers.sample import GaussianSampleLayer, BernoulliSampleLayer, GMSampleLayer
from ..layers.ar import ARConv2DLayer
from ..blocks.MADE import MADE
from ..layers.distributions import ClampLogVarLayer, MergeMeanLayer, MergeLogVarLayer, GaussianLikelihoodLayer, BernoulliLikelihoodLayer, GMLikelihoodLayer, FourierGaussianLikelihoodLayer, KLLayer, KLLayerGaussian, KLLayerGaussianMixture
from ..layers.merge import CondConcatLayer
from ..distributions import kl_normal2_normal2, log_normal2, log_bernoulli

from ..layers.transform import RFFTLayer, iRFFTLayer
from ..blocks.deconvolution import deconvolution

import math

class ladder_step():
    
    __metaclass__ = ABCMeta
    """
    Base element of a Ladder
    """

    @abstractmethod
    def connect_upward(self, d, y, rng=None):
        """
        Connects the bottom-up part of the ladder
        """
        self.d_in = d


    @abstractmethod
    def connect_downward(self, p, y, rng=None):
        """
        Connects the top-down part of the ladder
        """
        self.log_pz = None
        self.log_qz = None
        self.KL_term = None
        self.top_down_net = p 


class input_step(ladder_step):
    
    def __init__(self, n_filters=None, filter_size=5, downsample=False, output_nonlinearity=identity):
        """
            Filter bank for input and output, all of this is optional
        """
        self.n_filters = n_filters
        self.downsample = downsample
        self.output_nonlinearity = output_nonlinearity
        self.filter_size = filter_size
        
    def connect_upward(self, d, y, rng=None):
        self.d_in = d
        self.n_x = get_output_shape(d)[2]    
        self.n_filters_in = get_output_shape(d)[1]    
        stride = 2 if self.downsample else 1
        
        input_net = d
        if self.n_filters is not None:
            input_net = Conv2DLayer(input_net, num_filters=self.n_filters, filter_size=self.filter_size, stride=stride, nonlinearity=elu, pad='same')
        elif self.downsample:
            input_net = Conv2DLayer(input_net, num_filters=self.n_filters_in, filter_size=self.filter_size, stride=stride, nonlinearity=elu, pad='same')
        
        self.bottom_up_net = input_net
        
        return self.bottom_up_net

            
    def connect_downward(self, p, y, rng=None):
        # Deterministic layer:
        self.log_pz = None
        self.pz_smpl = None
        self.KL_term = None
        self.log_qz = None
        self.qz_smpl = None
        
        stride = 2 if self.downsample else 1
        input_net = p
        if self.downsample:
            input_net = deconvolution(input_net, num_filters=self.n_filters_in,
                                      stride=stride, nonlinearity=self.output_nonlinearity)
        else:
            input_net = Conv2DLayer(input_net, num_filters=self.n_filters_in, filter_size=self.filter_size, nonlinearity=self.output_nonlinearity, pad='same')

        self.top_down_net = input_net

        return self.top_down_net


    def GaussianLikelihood(self, input_logvar, diagCovariance=True):
        """
        This function returns a layers computing a Gaussian likelihood for the model,
        under the assumption that the noise is uncorrelated
        """
        if diagCovariance:
            return GaussianLikelihoodLayer(z=self.d_in, mean=self.top_down_net, log_var=input_logvar)
        else:
            out = RFFTLayer(self.top_down_net)
            z = RFFTLayer(self.d_in)
            return FourierGaussianLikelihoodLayer(z=z, mean=out, log_var=input_logvar, normalise=False)

    def BernoulliLikelihood(self):
        """
        returns a Bernoulli likelihood, WARNING: Make sure the output is between 0 and 1
        """
        return BernoulliLikelihoodLayer(self.d_in, self.top_down_net)


class resnet_step(ladder_step):

    def __init__(self, n_filters=32, latent_dim=0, IAF_sizes=[], downsample=False,  output_nonlinearity=identity):
        """
        Initialise the step
        """
        self.n_filters = n_filters
        self.latent_dim = latent_dim
        self.latent_dim = latent_dim
        self.downsample = downsample
        self.IAF_sizes = IAF_sizes
        self.output_nonlinearity = output_nonlinearity


    def connect_upward(self, d, y, rng=None):
        he_norm = HeNormal(gain='relu')
        self.d_in = d
        
        # Get the dimension of the input and check if downsampling is requested 
        self.n_filters_in = get_output_shape(d)[1]
        self.n_x =  get_output_shape(d)[2]
        stride = 2 if self.downsample else 1

        # If the input needs to be reshaped in any way we do it first
        if self.n_filters_in != self.n_filters or self.downsample:
            input_net = NonlinearityLayer(BatchNormLayer(d), elu)
            branch = input_net
        else:
            input_net = d
            branch = d
            branch = NonlinearityLayer(BatchNormLayer(branch), elu)

        #
        # Main branch
        #
        branch = Conv2DLayer(branch, num_filters=self.n_filters+self.latent_dim, filter_size=3, stride=1, nonlinearity=identity, pad='same', W=he_norm)

        if self.latent_dim > 0:
            branch_posterior = SliceLayer(branch, indices=slice(-2*self.latent_dim, None), axis=1)
            # Encoding for the posterior
            self.d_mu = SliceLayer(branch_posterior, indices=slice(0,self.latent_dim), axis=1)
            self.d_logvar = ClampLogVarLayer(SliceLayer(branch_posterior, indices=slice(self.latent_dim, None), axis=1))

            branch = SliceLayer(branch, indices=slice(0,-2*self.latent_dim), axis=1)

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
        
        return self.bottom_up_net


    def connect_downward(self, p, y, rng):
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
                branch_posterior = deconvolution(branch, num_filters=2*self.latent_dim,
                                                 stride=stride, nonlinearity=identity)
            else:
                branch_posterior = Conv2DLayer(branch,
                                            num_filters=2*self.latent_dim,
                                            filter_size=3, 
                                            stride=stride,
                                            nonlinearity=identity,
                                            pad='same',
                                            W=he_norm)
            self.tz_mu = SliceLayer(branch_posterior, indices=slice(0,self.latent_dim), axis=1)
            self.tz_logvar = ClampLogVarLayer(SliceLayer(branch_posterior, indices=slice(-self.latent_dim, None), axis=1))

            # Combine top-down inference information
            self.qz_mu = MergeMeanLayer(self.d_mu, self.d_logvar, self.tz_mu, self.tz_logvar)
            self.qz_logvar = MergeLogVarLayer(self.d_logvar, self.tz_logvar)
            
            # Sample at the origin of the IAF chain
            self.qz0 = GaussianSampleLayer(mean=self.qz_mu, log_var=self.qz_logvar, rng=rng, name='qz')
            
            # Add the Autoregressive layers
            z = self.qz0
            for i,h_size in enumerate(self.IAF_sizes):
                flipmask = ((i % 2) == 1)
                m = z
                for j in h_size:
                    m = ARConv2DLayer(m, num_filters=j, filter_size=3, pad='same', nonlinearity=elu, flipmask=flipmask)
                m = ARConv2DLayer(m, num_filters=self.latent_dim, filter_size=3, pad='same', nonlinearity=identity, flipmask=flipmask)
                
                # Only parametrising the mean, following recommendations from Kingma
                z = ElemwiseSumLayer([z, m])
                
            self.qz_smpl = z
            self.log_qz = GaussianLikelihoodLayer(z=self.qz0, mean=self.qz_mu, log_var=self.qz_logvar)
        else:  
            self.log_qz = None
            self.qz_smpl = None

        #
        # Main branch
        # 
        if self.downsample :
            branch = deconvolution(branch, num_filters=self.n_filters-self.latent_dim,
                                            stride=stride, nonlinearity=identity)
        else:
            branch = Conv2DLayer(branch,
                                num_filters=self.n_filters-self.latent_dim,
                                filter_size=3, 
                                stride=stride,
                                nonlinearity=identity,
                                pad='same',
                                W=he_norm)
        if self.latent_dim > 0:
            # Define step prior
            branch_prior = SliceLayer(branch, indices=slice(-2*self.latent_dim, None), axis=1)
            
            # Mixing with info from the top-down posterior branch
            branch_prior = ConcatLayer( [NonlinearityLayer(branch_prior, elu), 
                                         NonlinearityLayer(branch_posterior, elu)] )
            branch_prior = Conv2DLayer(branch_prior,
                                num_filters=2*self.latent_dim,
                                filter_size=3, 
                                nonlinearity=identity,
                                pad='same',
                                W=he_norm)
            self.pz_mu = SliceLayer(branch_prior, indices=slice(0,self.latent_dim), axis=1)
            self.pz_logvar = ClampLogVarLayer(SliceLayer(branch_prior, indices=slice(-self.latent_dim, None), axis=1))
            self.pz_smpl = GaussianSampleLayer(mean=self.pz_mu, log_var=self.pz_logvar, rng=rng, name='pz')
            self.log_pz = GaussianLikelihoodLayer(z=self.qz_smpl, mean=self.pz_mu, log_var=self.pz_logvar)
            
            # If the IAF is not used, evaluate the KL divergence analytically
            if len(self.IAF_sizes) == 0:
                self.KL_term = KLLayerGaussian(self.qz_mu, self.qz_logvar, self.pz_mu, self.pz_logvar)
            else:
                self.KL_term = KLLayer(self.log_pz, self.log_qz, factor=10.)
            #branch = SliceLayer(branch, indices=slice(0,-2*self.latent_dim), axis=1)
            ## Merge samples from the posterior into the main branch
            branch = CondConcatLayer(branch, self.qz_smpl, self.pz_smpl)
        else:
            self.log_pz = None
            self.pz_smpl = None
            self.KL_term = None

        branch = NonlinearityLayer(BatchNormLayer(branch), elu)
        branch = Conv2DLayer(branch,
                             num_filters=self.n_filters_in,
                             filter_size=3, 
                             stride=1,
                             nonlinearity=identity,
                             pad='same',
                             W=he_norm)

        if self.n_filters_in != self.n_filters or self.downsample:
            shortcut = deconvolution(input_net, num_filters=self.n_filters_in, nonlinearity=identity,  stride=stride)
        else:
            shortcut = input_net

        net = ElemwiseSumLayer([branch, shortcut])

        self.top_down_net = net
        return self.top_down_net



class dens_step(ladder_step):

    def __init__(self, n_units=128, nonlinearity=elu):
        """

        """
        self.n_units = n_units
        self.nonlinearity = nonlinearity
    
    def connect_upward(self, d, y, rng=None):
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


    def connect_downward(self, p, y, rng=None):
        he_norm = HeNormal(gain='relu')

        self.qz_smpl = None
        self.pz_smpl = None
        self.log_qz = None
        self.log_pz = None
        self.KL_term = None
        
        # If the input needs to be reshaped in any way we do it first
        branch = ConcatLayer([FlattenLayer(p), y])

        #
        # Main branch
        # 
        branch = batch_norm(DenseLayer(branch, num_units=self.n_units, nonlinearity=elu, W=he_norm))
        branch = batch_norm(DenseLayer(branch, num_units=self.n_units_in, nonlinearity=elu, W=he_norm))
        
        self.top_down_net = ReshapeLayer(branch,  shape=self.input_shape)
        return self.top_down_net


class gmm_prior_step(ladder_step):
    """
    Defines a prior using a Gaussian Mixture Model
    """
    
    def __init__(self,
                 n_units=[128, 128],
                 n_hidden=6,
                 n_gaussians=512,
                 IAF_sizes=[],
                 nonlinearity=elu, 
                 apply_nonlinearity=False):
        self.n_gaussians = n_gaussians
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.nonlinearity = nonlinearity
        self.IAF_sizes = IAF_sizes
        self.apply_nonlinearity = apply_nonlinearity

    def connect_upward(self, d, y, rng):
        """
        Computes the posterior approximation and sample from it
        """
        self.d_in = d

        if self.apply_nonlinearity:
            input_net = NonlinearityLayer(BatchNormLayer(d), elu)
        else:
            input_net = d
            
        self.qz_mu = DenseLayer(input_net, num_units=self.n_hidden, nonlinearity=None)
        self.qz_logvar= ClampLogVarLayer(DenseLayer(input_net, num_units=self.n_hidden, nonlinearity=None))

        # Sample from the Gaussian distribution
        self.qz0 = GaussianSampleLayer(mean=self.qz_mu, log_var=self.qz_logvar, rng=rng, name='z0_sample')
        
        # Add the Autoregressive layers
        z = self.qz0
        for i,h_size in enumerate(self.IAF_sizes):
            m = MADE(z, h_size, output_nonlinearity=None).reset('Full', i).get_output_layer()
            
            # Only parametrising the mean, following recommendations from Kingma
            z = ElemwiseSumLayer([z, m])
            
        self.qz_smpl = z
        self.log_qz = GaussianLikelihoodLayer(z=self.qz0, mean=self.qz_mu, log_var=self.qz_logvar)
        
        return self.qz_smpl


    def connect_downward(self, p, y, rng):
        """
        Computes the conditional prior 
        """
    
        self.batch_size = get_output_shape(p)[0]

        network = y
        for i in range(len(self.n_units)):
            network = batch_norm(DenseLayer(network, num_units=self.n_units[i],
                                          nonlinearity=self.nonlinearity,
                                          name="prior_%d" % i))

        # Conditional prior distribution
        self.pz_mu = ReshapeLayer(DenseLayer( network,
                                                 num_units=self.n_hidden*self.n_gaussians,
                                                 nonlinearity=identity,
                                                 name='pz_mu'),
                                     shape=(self.batch_size, self.n_hidden, self.n_gaussians))

        self.pz_logvar = ClampLogVarLayer(ReshapeLayer(DenseLayer( network,
                                                      num_units=self.n_hidden*self.n_gaussians,
                                                      nonlinearity=identity, 
                                                      name='pz_log_var'),
                                            shape=(self.batch_size, self.n_hidden, self.n_gaussians)))

        self.pz_w = DenseLayer(network, num_units=self.n_gaussians, nonlinearity=softmax, name='pw_log_var')

        # Prior samples
        self.pz_smpl = GMSampleLayer(mean=self.pz_mu, log_var=self.pz_logvar, weight=self.pz_w, rng=rng, name='pz')
        
        self.log_pz = GMLikelihoodLayer(z=p, mean=self.pz_mu, log_var=self.pz_logvar, weight=self.pz_w)
        
        # If the IAF is not used, evaluate the KL divergence analytically
        if len(self.IAF_sizes) == 0:
            self.KL_term = KLLayerGaussianMixture(self.qz_mu, self.qz_logvar, self.pz_mu, self.pz_logvar, self.pz_w)
        else:
            self.KL_term = KLLayer(self.log_pz, self.log_qz)
        self.top_down_net = p
        return self.top_down_net


    
