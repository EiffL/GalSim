# This module trains a GAN to sample from estimated posterior distribution
# instead of the VAE prior (so called realism constraint)
# See this paper (arXiv:1711.05772)
import numpy as np
from numpy.random import randn, randint

import theano
import theano.tensor as T

from lasagne.utils import unique
from lasagne.utils import floatX
from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, ReshapeLayer, FlattenLayer, NonlinearityLayer, get_output, get_all_params, ElemwiseSumLayer, get_output_shape, get_all_layers, batch_norm
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity, softmax
from lasagne.updates import adam, total_norm_constraint, apply_momentum, rmsprop, adamax
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import TransposedConv2DLayer, Upscale2DLayer
from lasagne.layers.dnn import Conv2DDNNLayer, Pool2DDNNLayer

class vaeGAN(object):

    def __init__(self, ladder_model, lambda_dist=0.1):
        """
        Initializes the model by loading a given ladder model and setting up
        a GAN to work in the latent space
        """
        self.ladder = ladder_model
        self.lambda_dist = lambda_dist
        self._c_in = T.matrix('c_in')
        self._a_in = T.matrix('a_in')
        self._sigma_q = T.vector('sigma_q')
        self._mu_q = T.vector('mu_q')

        self._build()

    def _build(self):
        """
        Builds the adversarial critic and amortized generator
        """

        # Extracts the sampling layers
        self._pz = self.ladder.prior_layer
        self._qz = self.ladder.code_layer
        l_x, l_y = self.ladder.input_layers
        x, y = self.ladder.input_variables

        #  Definition of the critic
        self._l_input_critic = InputLayer(shape=(self.batch_size, self.n_hidden),
                                          input_var=self._c_in, name='c_in')
        network = DenseLayer(self._l_input_critic, num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        c_out_layer = DenseLayer(network, num_units=1, nonlinearity=sigmoid)

        # Definition of the actor
        self._l_input_actor = InputLayer(shape=(self.batch_size,
                                                self.n_hidden),
                                         input_var=self._a_in, name='a_in')
        network = DenseLayer(self._l_input_actor, num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        gates = DenseLayer(network, num_units=self.n_hidden, nonlinearity=sigmoid)
        dz = DenseLayer(network, num_units=self.n_hidden, nonlinearity=identity)
        z_shift= ElemwiseMergeLayer([gates, ElemwiseSumLayer([self._l_input_actor, dz], coeffs=[-1,1])],
                  merge_function=T.mul)
        a_out_layer = ElemwiseSumLayer([self._l_input_actor, z_shift])

        # Creating the loss functions
        # First, critic loss
        c_prior = get_output(self.c_out_logits,
                             inputs={self._l_input_critic: self._pz, l_y: y})
        c_post  = get_output(self.c_out_logits,
                             inputs={self._l_input_critic: self._qz, l_y: y,
                             l_x: x})
        c_fake, zp, zt  = get_output([self.c_out_logits, self._pz, a_out_layer],
                                     inputs={self._l_input_critic: a_out_layer,
                                     l_y: y,
                                     self._l_input_actor: self._pz})

        ldist = 1. / self._sigma_q**2 * T.log(1 + (zp - zt)**2)

        loss_crit = 0.3*(binary_crossentropy(c_prior, T.zeros(c_prior.shape)) +
                         binary_crossentropy(c_post, T.ones(c_post.shape)) +
                         binary_crossentropy(c_fake, T.zeros(c_fake.shape)))

        loss_actor = 0.5*(binary_crossentropy(c_fake, T.ones(c_fake.shape)) +
                          self.lambda_dist * ldist)

        params_crit = get_all_params([c_out_layer], trainable=True)
        grads_crit = T.grad(loss_crit, params_crit)
        updates_crit = adam(grads_crit, params_crit,
                            learning_rate=self.learning_rate)

        params_actor = get_all_params([a_out_layer], trainable=True)
        grads_actor = T.grad(loss_actor, params_actor)
        updates_actor = adam(grads_actor, params_actor,
                             learning_rate=self.learning_rate)

        self._trainer_crit = theano.function([x, y], loss_crit,
                                             updates=updates_crit)
        self._trainer_actor = theano.function([y], loss_actor,
                                              updates=updates_actor)

        self._gen_sampl = theano.function([y], zt)
