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

class GenerativeGalaxyModel(object):
    """
    Generator object
    """

    _req_params = {'file_name': str}
    _opt_params = {}
    _single_params = []

    def __init__(self, dir=None, tags=None, session_config=None):
        """
        Initialisation of the generator, by loading a tensorflow model

        Parameters
        ----------
        dir: string
            Path to the tensorflow model to load
        """

        self.sess = tf.Session(config=session_config)

        # Load a saved model
        tf.saved_model.loader.load(sess, tags, export_dir=dir)

        # Add these quantities as required parameters for the sampling
        self.sample_req_params = {}
        self.sample_opt_params = {}
        self.sample_single_params = []

        for q in quantities:
            self.sample_req_params[q] = float

    def sample(self, catalog,  noise=None,  rng=None, x_interpolant=None, k_interpolant=None,
               pad_factor=4, noise_pad_size=0, gsparams=None):
        """
        Samples galaxy images from the model
        """
        # Build numpy array from data
        y = np.zeros((len(cat), len(self.quantities)))
        for j, q in enumerate(self.quantities):
            y[:, j] = cat[q]
        y = (y - self.y_shift) / self.y_scaling




    @classmethod
    def read(cls, file_name):
        """
        Reads in a pickled file and initialise object
        """
        f = open(file_name, 'rb')
        all_params = pickle.load(f)
        f.close()

        return cls(*all_params)
