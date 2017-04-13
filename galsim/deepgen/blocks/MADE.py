# This module contains the code for building a MADE block
import numpy as np

import theano
import theano.tensor as T

import lasagne
import lasagne.layers
from lasagne.layers import get_output_shape, get_output, DropoutLayer
from lasagne.nonlinearities import sigmoid, rectify

from ..layers.masked import MaskGenerator, MaskedLayer

class SeedGenerator(object):
    # This subclass purpose is to maximize randomness and still keep reproducibility

    def __init__(self, random_seed):
        self.rng = np.random.mtrand.RandomState(random_seed)

    def get(self):
        return self.rng.randint(42424242)

class MADE(object):
    """
    Object implementing a MADE along with utility functions
    """

    def __init__(self, incoming,  hidden_sizes,
                      nonlinearity=rectify,
                      output_nonlinearity=sigmoid,
                      dropout_rate=0,
                      mask_distribution=0,
                      use_cond_mask=False,
                      random_seed=1234):

        self.nonlinearity = nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.shuffled_once = False
        self.dropout_rate=dropout_rate
        self.seed_generator = SeedGenerator(random_seed)
        self.use_cond_mask = use_cond_mask

        input_size = get_output_shape(incoming)[-1]
        self.input_size = input_size

        self.x = T.matrix(name="in")

        # Initialize the masks
        self.mask_generator = MaskGenerator(input_size, hidden_sizes,
                                             mask_distribution,
                                             self.seed_generator.get())

        # Building structure of the MADE block
        network = incoming
        self.layers = []
        for i, n_h  in enumerate(hidden_sizes):
            layer = MaskedLayer(incoming=network,
                            num_units=n_h,
                            layerIdx=i,
                            mask_generator=self.mask_generator,
                            use_cond_mask=self.use_cond_mask,
                            nonlinearity=self.nonlinearity)

            network = DropoutLayer(layer, p = self.dropout_rate)
            self.layers.append(layer)

        # And the output layer
        outputLayerIdx = len(self.layers)
        network = MaskedLayer( incoming=network,
                               num_units=input_size,
                               layerIdx=outputLayerIdx,
                               mask_generator=self.mask_generator,
                               use_cond_mask=use_cond_mask,
                               nonlinearity=self.output_nonlinearity)
        self.layers.append(network)
        self.network = network
        # How to to shuffle weights
        masks_updates = [layer_mask_update for layer in self.layers for layer_mask_update in layer.shuffle_update]
        self.update_masks = theano.function(name='update_masks',
                                            inputs=[],
                                            updates=masks_updates)

        # Function to get output
        out = get_output(network, inputs={incoming : self.x}, deterministic=True)
        self.predict = theano.function([self.x], out)

        # Initialize the masks
        self.shuffle("Once")

    def get_output_layer(self):
        """
        Returns the output layer of the model
        """
        return self.network

    def shuffle(self, shuffling_type):
        if shuffling_type == "Once" and self.shuffled_once is False:
            self.mask_generator.shuffle_ordering()
            self.mask_generator.sample_connectivity()
            self.update_masks()
            self.shuffled_once = True
            return self

        if shuffling_type in ["Ordering", "Full"]:
            self.mask_generator.shuffle_ordering()
        if shuffling_type in ["Connectivity", "Full"]:
            self.mask_generator.sample_connectivity()
        self.update_masks()
        return self

    def reset(self, shuffling_type, last_shuffle=0):
        self.mask_generator.reset()

        # Always do a first shuffle so that the natural order does not gives us an edge
        self.shuffle("Full")

        # Set the mask to the requested shuffle
        for i in range(last_shuffle):
            self.shuffle(shuffling_type)

        return self

    def sample(self, nb_samples=1, mask_id=0):
        rng = np.random.mtrand.RandomState(self.seed_generator.get())

        self.reset("Full", mask_id)

        swap_order = self.mask_generator.ordering.get_value()

        samples = np.zeros((nb_samples, self.input_size), theano.config.floatX)

        for i in range(self.input_size):
            inv_swap = np.where(swap_order == i)[0][0]
            out = self.predict(samples)
            rng.binomial(p=out[:, inv_swap], n=1)
            samples[:, inv_swap] = rng.binomial(p=out[:, inv_swap], n=1)

        return samples
