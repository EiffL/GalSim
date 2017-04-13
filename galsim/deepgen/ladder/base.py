import sys
sys.setrecursionlimit(10000)

import theano
import theano.tensor as T

from lasagne.layers import get_output, get_all_params, get_output_shape
from lasagne.updates import adam, total_norm_constraint

from ..blocks.IAF import MADE_IAF

class ladder(object):

    def __init__(self, steps, prior, input_layers, input_variables,
                 hidden_sizes=[],
                 learning_rate=0.001):
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
        self.input_layers = input_layers
        self.input_variables = input_variables
        self.hidden_sizes = hidden_sizes

        self._z = T.matrix('z')
        self._lr = T.scalar('lr')

    def _build(self):
        """
        Builds the ladder from a list of steps
        """
        l_x, l_y = self.input_layers
        x, y = self.input_variables
        self.inputs = {l_x: x, l_y: y}

        ### Build and connect network
        # Connect the deterministic upward pass
        d_mu = l_x
        print "input shape", get_output_shape(d_mu)
        for s in self.steps:
            d_mu = s.connect_upward(d_mu, l_y)
            print "bottom-up", get_output_shape(d_mu)

        # Use an IAF parametrisation for the code
        self.code_layer, self.logqz = MADE_IAF(self.steps[-1].d_mu,
                                               self.steps[-1].d_logvar,
                                               self.hidden_sizes,
                                               self.inputs)

        # Connect the probabilistic downward pass
        qz_smpl = self.code_layer
        pz_smpl = self.code_layer

        for i, s in enumerate(self.steps[::-1]):
            if i == 0:
                pz_smpl, qz_mu, qz_logvar = s.connect_downward(pz_smpl, l_y, qz_smpl=qz_smpl)
            else:
                pz_smpl, qz_mu, qz_logvar = s.connect_downward(pz_smpl, l_y, tz_mu=qz_mu, tz_logvar=qz_logvar)
            print "top-down", get_output_shape(pz_smpl)
        # Changing output layer to the mean, for a Gaussian model where the noise is known
        self.output_layer = self.steps[0].p_mu

        # Connect prior network
        self.prior_layer = self.prior.connect(l_y)

        ### Compute the cost function
        # KL divergence of the code
        kl_prior = self.prior.kl_IAF(self.inputs, self.code_layer, self.logqz)

        # KL divergence of each steps
        kl_steps = []
        for s1, s2 in zip(self.steps[:-1], self.steps[1:]):
            kl_steps.append(s1.kl_normal(self.inputs, s2))

        # Finally, cost function of the reconstruction error
        log_px = self.steps[0].log_likelihood(self.inputs)

        # Total cost function
        LL = kl_prior - log_px
        for kl in kl_steps:
            LL += kl

        # Averaging over mini-batch
        LL = LL.mean()

        # Get trainable parameters and generate updates
        params = get_all_params([self.output_layer,
                                 self.steps[0].qz_smpl,
                                 self.prior_layer],
                                trainable=True)

        grads = T.grad(LL, params)
        #max_norm = 1.
        #mgrads = total_norm_constraint(grads, max_norm=max_norm)
        #clip_grad = 1.
        #cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
        updates = adam(grads, params, learning_rate=self._lr)

        self._trainer = theano.function([x, y, self._lr], [LL, log_px.mean(), kl_prior.mean()], updates=updates)

        # Get outputs from the recognition network
        z_smpl = get_output(self.code_layer, inputs=self.inputs, deterministic=True)
        self._code_sampler = theano.function([x, y], z_smpl)

        # Get outputs from the prior network
        pz_smpl = get_output(self.prior_layer, inputs={l_y: y}, deterministic=True)
        self._prior_sampler = theano.function([y], pz_smpl)

        # Get outputs from the generative network
        x_smpl = get_output(self.output_layer, inputs={self.code_layer: self._z, l_y: y}, deterministic=True)
        self._output_sampler = theano.function([self._z, y], x_smpl)

        # Get outputs from the generative network mean
        x_smpl_mean = get_output(self.steps[0].p_mu, inputs={self.code_layer: self._z, l_y: y}, deterministic=True)
        self._output_sampler_mean = theano.function([self._z, y], x_smpl_mean)
