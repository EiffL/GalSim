
import theano
import theano.tensor as T

from lasagne.layers import DenseLayer, InputLayer, ConcatLayer, ReshapeLayer, FlattenLayer, NonlinearityLayer, get_output, get_all_params, batch_norm, get_output_shape
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity, softmax
from ..layers.sample import GaussianSampleLayer, GMSampleLayer
from ..layers.transform import ClampLogVarLayer
from lasagne.init import GlorotUniform, Constant

from ..distributions import kl_normal2_gm2, log_gm2

class ladder_prior(object):
    """
    Defines a specific prior to use for the code in the ladder
    """

    def __init__(self):
        """
        Do some initialisation
        """
        pass

    def connect(self, cond_layer):
        """
        Creates and connect the architecture of the prior
        network, returns the sampling layer
        """
        pass

    def kl_normal(self, inputs, top_step):
        """
        Computes the kl divergence between prior and
        posterior using analytical expression for Gaussian
        distribution
        """
        pass

    def kl_IAF(self, inputs, l_z, log_qz_given_x):
        """
        Computes the KL divergence using an IAF parametrised
        posterior
        """
        pass


class gmm_prior(ladder_prior):
    """
    Defines a parametrised prior using a GMM
    """
    def __init__(self,
                 n_units=[128, 128],
                 n_hidden=6,
                 n_gaussians=512,
                 nonlinearity=elu):
        self.n_gaussians = n_gaussians
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.nonlinearity = nonlinearity

    def connect(self, input_conditional_layer):
        """
        Builds the network and returns sampling layer
        """
        self._l_input = input_conditional_layer
        self.batch_size = get_output_shape(input_conditional_layer)[0]

        network = self._l_input
        for i in range(len(self.n_units)):
            network = DenseLayer(network, num_units=self.n_units[i],
                                          nonlinearity=self.nonlinearity,
                                          W=GlorotUniform(),
                                          name="prior_%d" % i)

        # Conditional prior distribution
        self._l_pz_mu = ReshapeLayer(DenseLayer( network,
                                                 num_units=self.n_hidden*self.n_gaussians,
                                                 nonlinearity=identity, W=Constant(),
                                                 name='pz_mu'),
                                     shape=(self.batch_size, self.n_hidden, self.n_gaussians))

        self._l_pz_log_var = ClampLogVarLayer(ReshapeLayer(DenseLayer( network,
                                                      num_units=self.n_hidden*self.n_gaussians,
                                                      nonlinearity=identity, W=Constant(),
                                                      name='pz_log_var'),
                                            shape=(self.batch_size, self.n_hidden, self.n_gaussians)))

        self._l_pw = DenseLayer(network, num_units=self.n_gaussians, nonlinearity=softmax, name='pw_log_var')

        # Prior samples
        self._l_pz_smpl = GMSampleLayer(mean=self._l_pz_mu, log_var=self._l_pz_log_var,
                                         weight=self._l_pw, name='pz')

        # Storing output variables
        return self._l_pz_smpl


    def kl_normal(self, inputs, top_step):
        """
        Get the KL divergence for this prior
        """
        z_mu, z_log_var, pz_mu, pz_log_var, pz_w = get_output([top_step.d_mu, top_step.d_logvar,
                                      self._l_pz_mu, self._l_pz_log_var, self._l_pw], inputs=inputs)

        shape = get_output_shape(top_step.d_mu)

        return kl_normal2_gm2(z_mu, z_log_var, pz_mu, pz_log_var, pz_w, eps=1e-6).clip(0.125,100).sum(axis=-1)

    def kl_IAF(self, inputs, l_z, log_qz_given_x):
        """
        Get the KL divergence under an IAF parametrised posterior
        """
        z, pz_mu, pz_log_var, pz_w = get_output([l_z, self._l_pz_mu,
                                                 self._l_pz_log_var,
                                                 self._l_pw], inputs=inputs)
        log_pz = log_gm2(z, mean=pz_mu, log_var=pz_log_var, weights=pz_w)
        return log_qz_given_x - log_pz
