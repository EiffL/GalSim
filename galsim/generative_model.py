# Copyright (c) 2012-2017 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""@file generative_model.py
Functions defining a light profile produced by a deep generative model

TODO: add more documentation
"""
import galsim
import tensorflow as tf
import numpy as np

class GenerativeGalaxyModel(object):
    """
    Generator object
    """

    _req_params = {'file_name': str}
    _opt_params = {}
    _single_params = []

    def __init__(self, export_dir=None,
                tags=['serve'],
                signature_name='serving_default',
                output_name='SBProfile',
                session_config=None):
        """
        Initialisation of the generator, by loading a tensorflow model

        Parameters
        ----------
        dir: string
            Path to the tensorflow model to load
        """

        # Create new tensorflow session
        self.sess = tf.Session(config=session_config)

        # Load the saved model in session
        self.graph_def = tf.saved_model.loader.load(self.sess, tags=tags, export_dir=export_dir)
        self.graph = tf.get_default_graph()

        # Extracts the signature of requested function
        signature = self.graph_def.signature_def[signature_name]

        # get the function output
        self.output = signature.outputs[output_name]

        # Add these quantities as required parameters for the sampling
        self.sample_req_params = {}
        self.sample_opt_params = {}
        self.sample_single_params = []

    def sample(self, n_samples,  noise=None,  rng=None, x_interpolant=None, k_interpolant=None,
               pad_factor=4, noise_pad_size=0, gsparams=None):
        """
        Samples galaxy images from the model
        """

        # Run the graph
        x = self.sess.run(self.output.name)

        # Now, we build an InterpolatedImage for each of these
        ims = []
        for i in range(len(x)):
            im = galsim.Image(np.ascontiguousarray(x[i].reshape((28,28)).astype(np.float64)),
                              scale=0.03)
            ims.append(galsim.InterpolatedImage(im,
                                                x_interpolant=x_interpolant,
                                                k_interpolant=k_interpolant,
                                                pad_factor=pad_factor,
                                                noise_pad_size=noise_pad_size,
                                                noise_pad=noise,
                                                rng=rng,
                                                gsparams=gsparams))
        return ims


    @classmethod
    def read(cls, file_name):
        """
        Reads in a pickled file and initialise object
        """
        f = open(file_name, 'rb')
        all_params = pickle.load(f)
        f.close()

        return cls(*all_params)
