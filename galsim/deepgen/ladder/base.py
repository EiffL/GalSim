import sys
sys.setrecursionlimit(10000)

import theano
import theano.tensor as T

import numpy as np

from lasagne.layers import get_output, get_all_params, get_output_shape, InputLayer, ElemwiseSumLayer
from lasagne.updates import adam, total_norm_constraint
from lasagne.regularization import regularize_network_params, l2
from lasagne.utils import floatX

from ..layers.distributions import LogNormalLayer




class ladder(object):

    def __init__(self, n_c, n_x, n_y,
                 steps,
                 batch_size=32,
                 noise_std=1,
                 learning_rate=0.001):
        """
        Initialises the ladder structure

        Parameters
        ----------
        steps: list of ladder_steps
            List of steps for the ladder in bottom up order
        """
        self.steps = steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_c = n_c
        self.n_x = n_x
        self.n_y = n_y
        
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

        # The last step of the ladder should be the posterior/prior element
        self.code_layer = d_mu

        # Connect the probabilistic downward pass
        p = self.code_layer
        for i, s in enumerate(self.steps[::-1]):
            p = s.connect_downward(p, self.l_y)
            print("top-down", get_output_shape(p))
            
        # Output of the ladder
        self.output_layer = p
        
        # Get outputs from the generative network for a given code
        ins = {self.l_x: self._x, self.code_layer: self._z, self.l_y: self._y}
        x_smpl = get_output(self.output_layer, inputs=ins, deterministic=True, alternative_path=True)
        self._decoder = theano.function([ self._x, self._z, self._y], x_smpl)
        print "success"
        
        # Building the cost function
        self.cost_layers = [LogNormalLayer(z=self.l_x, mean=self.output_layer, cst_std=noise_std),]
        for s in self.steps:
            if s.KL_term is not None:
                self.cost_layers.append(s.KL_term)

        # Summing terms
        self.cost_layer = ElemwiseSumLayer(self.cost_layers)
        
        LL, log_px, kl0, kl1, klp = get_output([self.cost_layer, self.cost_layers[0], self.cost_layers[1], self.cost_layers[2], self.cost_layers[-1]], inputs=self.inputs)

        # Get trainable parameters and generate updates
        params = get_all_params([self.cost_layer], trainable=True)

        grads = T.grad(- LL.mean(), params)
        #clip_grad = 10.
        #cgrads = [T.clip(g, -clip_grad, clip_grad) for g in grads]
        updates = adam(grads, params, learning_rate=self._lr)
        
        # Training function
        self._trainer = theano.function([self._x, self._y, self._lr], [-LL.mean(), log_px.mean(), kl0.mean(),kl1.mean(), klp.mean()], updates=updates)

        
        # Randomly samples from model
        #ins = {self.l_y: self._y}
        #for s in self.steps[::-1]:
            #if s.qz_smpl is not None:
                #ins[s.qz_smpl] = get_output(s.pz_smpl, inputs=ins, deterministic=True)
        #x_smpl_p = get_output(self.output_layer, inputs=ins, deterministic=True)
        #self._sampler = theano.function([self._y], x_smpl_p)

        # Get outputs from the recognition network
        z_smpl = get_output(self.code_layer, inputs=self.inputs, deterministic=True)
        self._code_sampler = theano.function([self._x, self._y], z_smpl)

        # Get outputs from the prior network
        pz_smpl = get_output(self.steps[-1].pz_smpl, inputs={self.l_y: self._y}, deterministic=True)
        self._prior_sampler = theano.function([self._y], pz_smpl)
        
        # Get reconstruction of auto-encoder
        ins = {self.l_x: self._x, self.l_y: self._y}
        x_rec = get_output(self.output_layer, inputs=ins, deterministic=True)
        self._reconstruct = theano.function([self._x, self._y], x_rec)

        
    
    def transform(self, X, y):
        """
        Transforms the data into latent code.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_features)
            The data to be transformed.

        y: array_like (n_samples, n_conditional_features)
            Conditional data.

        Returns
        -------
        h: array of shape (n_samples, n_hidden)
            Latent representation of the data.
        """

        res = []
        nsamples  = X.shape[0]

        # Process data using batches, for optimisation and memory constraints
        for i in range(int(nsamples/self.batch_size)):
            z = self._code_sampler(floatX(X[i*self.batch_size:(i+1)*self.batch_size]),
                                   floatX(y[i*self.batch_size:(i+1)*self.batch_size]))
            res.append(z)

        if nsamples % (self.batch_size) > 0 :
            i = int(nsamples/self.batch_size)
            ni = nsamples % (self.batch_size)
            xdata = np.zeros((self.batch_size,) + X.shape[1:])
            xdata[:ni] = X[i*self.batch_size:]
            ydata = np.zeros((self.batch_size, y.shape[1]))
            ydata[:ni] = y[i*self.batch_size:]
            z = self._code_sampler(floatX(xdata), floatX(ydata))

            res.append(z[:ni])

        # Concatenate processed data
        z = np.concatenate(res)
        return z

    def inverse_transform(self, h, y):
        """
        Decodes a latent code.

        Parameters
        ----------
        h: array_like of shape (n_samples, n_hidden)
            Latent code

        Y: array_like of shape (n_samples, n_conditional_features)
            Conditional variables that conditioned the code

        mean: boolean, optional
            If True, returns the mean of the output distribution, otherwise
            returns a sample from this distribution. (default: False)

        Returns
        -------
        X: array, of shape (n_samples, n_features)
            Reconstruction
        """

        res = []
        n_samples  = h.shape[0]
        
        sampler = self._decoder

        # Process data using batches, for optimisation and memory constraints
        for i in range(int(n_samples/self.batch_size)):
            X = sampler(floatX(np.zeros((self.batch_size,self.n_c,self.n_x,self.n_x))), floatX(h[i*self.batch_size:(i+1)*self.batch_size]),
                        floatX(y[i*self.batch_size:(i+1)*self.batch_size]))
            res.append(X)

        if n_samples % (self.batch_size) > 0 :
            i = int(n_samples/self.batch_size)
            ni = n_samples % (self.batch_size)
            hdata = np.zeros((self.batch_size, h.shape[1]))
            hdata[:ni] = h[i*self.batch_size:]
            ydata = np.zeros((self.batch_size, y.shape[1]))
            ydata[:ni] = y[i*self.batch_size:]
            X = sampler(floatX(np.zeros((self.batch_size,self.n_c,self.n_x,self.n_x))),floatX(hdata),  floatX(ydata))
                
            res.append(X[:ni])

        # Concatenate processed data
        X = np.concatenate(res)
        return X

    def sample_prior(self, y=None, n_samples=None):
        """
        Draws samples from the prior distribution conditioned by y.

        Parameters
        ----------
        y: array, of shape (n_samples, n_conditional_features)
            Conditional variable used by the prior.

        n_samples: int
            Number of samples to draw.

        Returns
        -------
        h: array, of shape (n_samples, n_hidden)
            Randomly drawn code samples.
        """

        res = []
        if n_samples is not None:
            nsamples = min([y.shape[0], n_samples])
        else:
            nsamples = y.shape[0]

        # Process data using batches, for optimisation and memory constraints
        for i in range(int(nsamples/self.batch_size)):
            z = self._prior_sampler(floatX(y[i*self.batch_size:(i+1)*self.batch_size]))
            res.append(z)

        if nsamples % (self.batch_size) > 0 :
            i = int(nsamples/self.batch_size)
            ni = nsamples % (self.batch_size)
            ydata = np.zeros((self.batch_size, y.shape[1]))
            ydata[:ni] = y[i*self.batch_size:]
            z = self._prior_sampler(floatX(ydata))
            res.append(z[:ni])

        # Concatenate processed data
        z = np.concatenate(res)
        return z

    def sample(self, y=None, n_samples=None):
        """
        Draws new samples from the generative network and the prior distribution, conditioned
        on variable y.

        Parameters
        ----------
        y: array, of shape (n_samples, n_conditional_features)
            Conditional variable used by the prior.
            
        n_samples: int, optional
            Number of samples to draw.
    
        mean: boolean, optional
            If True, returns the mean of the output distribution, otherwise
            returns a sample from this distribution. (default: False)

        Returns
        -------
        X: array, of shape (n_samples, n_features)
            Randomly drawn samples.
        """
        res = []
        if n_samples is not None:
            nsamples = min([y.shape[0], n_samples])
        else:
            nsamples = y.shape[0]
        
        sampler = self._sampler

        # Process data using batches, for optimisation and memory constraints
        for i in range(int(n_samples/self.batch_size)):
            X = sampler(floatX(y[i*self.batch_size:(i+1)*self.batch_size]))
            res.append(X)

        if n_samples % (self.batch_size) > 0 :
            i = int(n_samples/self.batch_size)
            ni = n_samples % (self.batch_size)
            ydata = np.zeros((self.batch_size, y.shape[1]))
            ydata[:ni] = y[i*self.batch_size:]
            X = sampler(floatX(ydata))
                
            res.append(X[:ni])

        # Concatenate processed data
        X = np.concatenate(res)
        return X
