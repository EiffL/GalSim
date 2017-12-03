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
from lasagne.layers import TransposedConv2DLayer, Upscale2DLayer, ElemwiseMergeLayer
from lasagne.layers.dnn import Conv2DDNNLayer, Pool2DDNNLayer
from theano.tensor.nnet import binary_crossentropy

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
        self.learning_rate= T.scalar('lr')
        self._eps = T.matrix('epsilon')
        self.batch_size = self.ladder.batch_size
        self.n_hidden = get_output_shape(self.ladder.code_layer)[1]

        self._build()

    def _build(self):
        """
        Builds the adversarial critic and amortized generator
        """

        # Extracts the sampling layers
        self._pz = FlattenLayer(self.ladder.steps[-1].pz_smpl)
        self._qz = self.ladder.code_layer
        l_x, l_y = (self.ladder.l_x,self.ladder.l_y)
        x, y = (self.ladder._x, self.ladder._y)

        #  Definition of the critic
        self._l_input_critic = InputLayer(shape=(self.batch_size, self.n_hidden),
                                          input_var=self._c_in, name='c_in')
        net_y   = batch_norm(DenseLayer(l_y, num_units=2048, nonlinearity=elu))
        network = DenseLayer(ConcatLayer([self._l_input_critic, net_y]), num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        c_out_layer = DenseLayer(network, num_units=1, nonlinearity=None)

        # Definition of the actor
        self._l_input_actor = InputLayer(shape=(self.batch_size,
                                                self.n_hidden),
                                         input_var=self._a_in, name='a_in')
        network = DenseLayer(self._l_input_actor, num_units=2048, nonlinearity=elu)
        net_y   = batch_norm(DenseLayer(l_y, num_units=2048, nonlinearity=elu))
        network = DenseLayer(ConcatLayer([self._l_input_actor, net_y]), num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        network = DenseLayer(network, num_units=2048, nonlinearity=elu)
        gates = DenseLayer(network, num_units=self.n_hidden, nonlinearity=sigmoid)
        dz = DenseLayer(network, num_units=self.n_hidden, nonlinearity=identity)
        z_shift= ElemwiseMergeLayer([gates, ElemwiseSumLayer([self._l_input_actor, dz], coeffs=[-1,1])],
                  merge_function=T.mul)
        a_out_layer = ElemwiseSumLayer([self._l_input_actor, z_shift])

        # Creating the loss functions
        # First, critic loss

        pz = get_output(self._pz, inputs={l_y: y})
        qz = get_output(self._qz, inputs={l_y: y, l_x: x})
        tz = get_output(a_out_layer, inputs={self._l_input_actor:pz})

        ldist = (1. / self._sigma_q**2 * T.log(1 + (pz - tz)**2)).mean()

        c_post = get_output(c_out_layer, inputs={self._l_input_critic:qz})
        c_fake = get_output(c_out_layer, inputs={self._l_input_critic:tz})

        mixed_x = (self._eps * tz) + (1-self._eps)*qz
        c_mixed = get_output(c_out_layer, inputs={self._l_input_critic:mixed_x})
        grad_mixed = T.grad(T.sum(c_mixed), mixed_x)
        norm_grad_mixed = T.sqrt(T.sum(T.square(grad_mixed), axis=-1))
        grad_penalty = T.mean(T.square(norm_grad_mixed - 1.))
        loss_c_real = c_post.mean()
        loss_c_fake = c_fake.mean()

        # Building cost functions for wGAN-GP
        loss_c = loss_c_fake - loss_c_real + 10. * grad_penalty
        loss_a = - loss_c_fake + self.lambda_dist * ldist

        params_crit = get_all_params([c_out_layer], trainable=True)
        updates_crit = adam(loss_c, params_crit,
                            learning_rate=self.learning_rate,
                            beta1=0.0, beta2=0.9)

        params_actor = get_all_params([a_out_layer], trainable=True)
        updates_actor = adam(loss_a, params_actor,
                             learning_rate=self.learning_rate,
                             beta1=0.0, beta2=0.9)

        self._trainer_crit = theano.function([x, y, self._eps, self.learning_rate], loss_c,
                                             updates=updates_crit)
        self._trainer_actor = theano.function([y, self._sigma_q, self.learning_rate], loss_a,
                                              updates=updates_actor)

        tzd = get_output(a_out_layer, inputs={self._l_input_actor:pz}, deterministic=True)
        self._gen_sampl = theano.function([y], tzd)
