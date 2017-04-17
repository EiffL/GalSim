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
Base class defining a generative model of galaxy images.

A generative model can be trained on some real astrophysical data and then used
to conditionally sample new galaxy images.

TODO: Add documentation.
"""
import galsim
import time
import numpy as np

from deepgen import ladder, gmm_prior, resnet_step, dens_step

from numpy.random import randn, randint
from lasagne.utils import floatX

class LadderVAE(GenerativeGalaxyModel):
    """
    Generative galaxy model based on a Ladder Variational AutoEncoder
    """
    def __init__(self,
                 stamp_size,
                 quantities=[],
                 batch_size=32,
                 n_hidden=128,
                 n_bands=1):
        super(self.__class__, self).__init__(quantities)
        self.stamp_size = stamp_size
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.n_bands = n_bands

        # Create the architecture of the ladder
        p = gmm_prior(n_units=[256, 512],
                      n_hidden=self.n_hidden,
                      n_gaussians=256)

        # First resnet layer, output 32x32x32
        resnet_1 = resnet_step(n_filters_in=self.n_bands,
                               n_filters=[16, 32],
                               latent_dim=32,
                               resnet_per_stage=3,
                               prefilter=True)
        # TODO: Fix that
        resnet_1.noise_std = 0.002  # Rough noise std deviation in the images

        # Second resnet layer, output 256x8x8
        resnet_2 = resnet_step(n_filters_in=32, n_filters=[64, 128, 256],
                               latent_dim=256, resnet_per_stage=2, prefilter=False)

        # Dense encoding layer
        dense_1 = dens_step(n_out= self.n_hidden,
                            n_units=[512, 256])


        # Build the ladder
        self.model = ladder(n_bands, stamp_size, len(quantities),
                            [resnet_1, resnet_2, dense_1], p,
                            learning_rate=0.001)

    def fit(self, X, features):
        """
        Train the model

        Parameters
        ----------

        X: list of GSObjects
            Images to use for training

        features: astropy Table with corresponding quantities
            table with features to train on
        """
        
        self.batch_per_epoch = len(x_train)/batch_size

        # Main training loop
        for i in range(60):
            if i % 20 == 0:
                learning_rate *= 0.1
            print(i, learning_rate)

            train_err = 0.
            train_kl = 0.
            train_logpx = 0.
            start_time = time.time()
            count = 0
            display = False
            old_logpx = 0

            tmp_train_err=0
            tmp_train_kl=0
            tmp_train_logpx=0
            tmp_count=0
            tmp_start_time = time.time()

            for b in range(batch_per_epoch):

                xdata, ydata = batches.next()

                # Train the auto-encoder
                ce_err, logpx_err, kl_err  = model._trainer(floatX(xdata),
                                                            floatX(ydata),
                                                            floatX(learning_rate))

                train_err += ce_err
                train_kl += kl_err
                train_logpx += logpx_err

                tmp_train_err += ce_err
                tmp_train_kl += kl_err
                tmp_train_logpx += logpx_err

                count += 1
                tmp_count+=1

                if tmp_count >= batch_print:
                    print("update took %f s, and epoch is done at %f percent"%(time.time() - tmp_start_time, 100.*b/batch_per_epoch))
                    print("Training loss: %f ; kl %f ; log_px_z %f"%(tmp_train_err / (tmp_count), tmp_train_kl / (tmp_count), tmp_train_logpx/ (tmp_count)))
                    tmp_train_err=0
                    tmp_train_kl=0
                    tmp_train_logpx=0
                    tmp_count=0
                    tmp_start_time = time.time()

            # Then we print the results for this epoch:
            print("Epoch took %f s"%(time.time() - start_time))
            print("Training loss: %f ; kl %f ; log_px_z %f"%(train_err / (count), train_kl / (count), train_logpx/ (count)))



    def sample(self, y):
        """
        Samples images from the model
        """
        pass
    
    
    
    def write(self):
        """
        Exports the  trainned  model
        """
        pass
