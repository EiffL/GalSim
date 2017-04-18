import sys
sys.setrecursionlimit(10000)

import theano
import theano.tensor as T

from lasagne.layers import get_output, get_all_params, get_output_shape, InputLayer
from lasagne.updates import adam, total_norm_constraint
from lasagne.regularization import regularize_network_params, l2

from ..blocks.IAF import MADE_IAF


class ladder(object):

    def __init__(self, n_c, n_x, n_y,
                 steps, prior,
                 batch_size=32,
                 IAF_size=[[128,128], [128,128]],
                 learning_rate=0.001,
                 l2_reg=1e-4):
        """
        Initialises the ladder structure

        Parameters
        ----------
        steps: list of ladder_steps
            List of steps for the ladder in bottom up order

        prior: ladder_prior
            Prior object to use at the top level of the ladder

        inputs: dict
            Dictionnary of the input layer and variables
        """
        self.steps = steps
        self.prior = prior
        self.learning_rate = learning_rate
        self.hidden_sizes = IAF_size
        self.batch_size = batch_size
        self.n_c = n_c
        self.n_x = n_x
        self.n_y = n_y
        self.l2_reg = l2_reg
        
        # Input variable
        self._x = T.tensor4('x')
        # Input condition
        self._y = T.matrix('y')
        # Code variable
        self._z = T.matrix('z')
        # Variable for the learning rate
        self._lr = T.scalar('lr')
        
        
        self.l_x = InputLayer(shape=(self.batch_size, self.n_c,
                                     self.n_x, self.n_x),
                              input_var=self._x, name='x')
        
        self.l_y = InputLayer(shape=(self.batch_size, self.n_y),
                              input_var=self._y, name='y')

        self.inputs = {self.l_x: self._x, self.l_y: self._y}

        ### Build and connect network
        # Connect the deterministic upward pass
        d_mu = self.l_x
        print("input shape", get_output_shape(d_mu))
        for s in self.steps:
            d_mu = s.connect_upward(d_mu, self.l_y)
            print("bottom-up", get_output_shape(d_mu))

        # Use an IAF parametrisation for the code
        self.code_layer, self.logqz = MADE_IAF(d_mu,
                                               self.hidden_sizes,
                                               self.inputs)

        # Connect the probabilistic downward pass
        qz_smpl = self.code_layer
        p = self.code_layer

        for i, s in enumerate(self.steps[::-1]):
            p = s.connect_downward(p, self.l_y)
            print("top-down", get_output_shape(p))
            
        # Changing output layer to the mean, for a Gaussian model where the noise is known
        self.output_layer = p

        # Connect prior network
        self.prior_layer = self.prior.connect(self.l_y)

        ### Compute the cost function
        # KL divergence of the code
        kl_prior = self.prior.kl_IAF(self.inputs, self.code_layer, self.logqz)

        # KL divergence of each steps
        kl_steps = []
        for s in self.steps:
            kl_steps.append(s.kl_normal(self.inputs))

        # Finally, cost function of the reconstruction error
        log_px = self.steps[0].log_likelihood(self.inputs)

        # Total cost function
        LL = kl_prior - log_px
        for kl in kl_steps:
            LL += kl

        # Averaging over mini-batch
        LL = LL.mean()

        # Get trainable parameters and generate updates
        params = get_all_params([self.output_layer, self.prior_layer, self.code_layer] + [s.pz_smpl for s in self.steps], trainable=True)

        grads = T.grad(LL, params)
        clip_grad = 10.
        cgrads = [T.clip(g, -clip_grad, clip_grad) for g in grads]
        updates = adam(cgrads, params, learning_rate=self._lr)

        self._trainer = theano.function([self._x, self._y, self._lr], [LL, log_px.mean(), kl_prior.mean(), self.logqz.mean()], updates=updates)

        # Get outputs from the recognition network
        z_smpl = get_output(self.code_layer, inputs=self.inputs, deterministic=True)
        self._code_sampler = theano.function([self._x, self._y], z_smpl)

        # Get outputs from the prior network
        pz_smpl = get_output(self.prior_layer, inputs={self.l_y: self._y}, deterministic=True)
        self._prior_sampler = theano.function([self._y], pz_smpl)

        # Get outputs from the generative network
        ins = {self.code_layer: self._z, self.l_y: self._y}
        for s in self.steps[::-1]:
            ins[s.qz_smpl] = get_output(s.pz_smpl, inputs=ins, deterministic=True)
        x_smpl = get_output(self.output_layer, inputs=ins,
                            deterministic=True)
        self._output_sampler = theano.function([self._z, self._y], x_smpl)


