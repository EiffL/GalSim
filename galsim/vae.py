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
from .generative_model import GenerativeGalaxyModel

from deepgen import ladder, gmm_prior_step, resnet_step, dens_step

from numpy.random import randint, permutation
from lasagne.utils import floatX


def _preprocessing_worker(params):
    """
    Outputs an image after applying random rotation, scaling and noise
    padding
    """
    real_galaxy_catalog, stamp_size, pixel_scale = params
    g_o = galsim.RealGalaxy(real_galaxy_catalog, noise_pad_size=stamp_size*pixel_scale, pad_factor=1)
    g = g_o.original_gal.rotate(galsim.Angle(-np.random.rand()*2*np.pi, galsim.radians)) 
    im = galsim.Image(stamp_size, stamp_size)
    g.drawImage(image=im, scale=pixel_scale)
    return im.array

class ResNetVAE(GenerativeGalaxyModel):
    """
    Generative galaxy model based on a Ladder Variational AutoEncoder
    """
    def __init__(self,
                 stamp_size,
                 quantities=[],
                 batch_size=32,
                 n_bands=1,
                 pixel_scale=0.03):
        super(self.__class__, self).__init__(quantities)
        self.stamp_size = stamp_size
        self.batch_size = batch_size
        self.n_bands = n_bands
        self.pixel_scale = pixel_scale

        n_hidden = 8

        # Create the architecture of the ladder
        p = gmm_prior_step(n_units=[256, 256],
                    n_hidden=n_hidden,
                    n_gaussians=512)

        # First resnet layer
        resnet_1 = resnet_step(n_filters=64,
                            latent_dim=16,
                            prefilter=True)

        resnet_2 = resnet_step(n_filters=64,
                            latent_dim=16,
                            downsample=True)

        #resnet_3 = resnet_step(n_filters=128,
                                #latent_dim=32,
                                #downsample=True)

        #resnet_4 = resnet_step(n_filters=128,
                                #latent_dim=32,
                                #downsample=True)

        dense_4 = dens_step(n_units=128)

        # Build the ladder
        self.model = ladder(n_bands, stamp_size,
                            len(quantities),
                            [resnet_1, resnet_2, dense_4, p],
                            batch_size=batch_size)


    def fit(self, real_cat, param_cat, valid_index=None,
            learning_rate=0.001, n_epochs=10, lr_step=5):
        """
        Train the model

        Parameters
        ----------
        
        real_cat
        
        param_cat
        
        """
        # multiprocessing for online generation of training images
        from multiprocessing import Pool
        pool = Pool()
        
        if valid_index is None:
            valid_index = range(real_cat.nobjects)

        ngal = len(valid_index)

        batch_per_epoch = ngal/self.batch_size

        xdata = np.zeros((self.batch_size, self.n_bands, self.stamp_size, self.stamp_size)).astype('float32')
        ydata = np.zeros((self.batch_size, len(self.quantities))).astype('float32')

        # Loop over training epochs
        for i in range(n_epochs):
            start_time = time.time()

            # Update learning rate
            if  i > 0 and (i % lr_step == 0):
                learning_rate /= 10.
                print("Updating learning rate to %f"%learning_rate)

            acc_train_err = 0
            acc_logpx = 0
            acc_klp = 0
            count = 0

            # Random indices to shuffle the data
            inds = permutation(ngal)

            # Construct the batch
            def get_batch_params(inds_batch):
                # Sets up parameters for the workers
                params = []
                for j in range(self.batch_size):
                    orig_index = inds_batch[j]
                    gal_image = real_cat.getGal(orig_index)
                    psf_image = real_cat.getPSF(orig_index)
                    noise_image, pixel_scale, var = real_cat.getNoiseProperties(orig_index)
                    params.append(((gal_image, psf_image, noise_image, pixel_scale, var),
                                   self.stamp_size, self.pixel_scale))
                return params
            
            # Loop over batches
            for b in range(batch_per_epoch):

                # Indices of objects to include in batch
                inds_batch = valid_index[inds[b*self.batch_size:(b+1)*self.batch_size]]
                
                if b == 0:
                    params = get_batch_params(inds_batch)
                    preprocessing_pool = pool.map_async(_preprocessing_worker, params)

                # recarrays are the absolute worst, can't find another
                # way to exctract the values than through a loop
                for j, q in  enumerate(self.quantities):
                    ydata[:,j] = param_cat[q][inds_batch]

                sigma_data = np.log(real_cat.variance[inds_batch])

                # Get the preprocessed images, fails after a minute
                ims = preprocessing_pool.get(timeout=60)
                xdata[:,0,:,:] = np.stack(ims)
                return xdata, ydata, sigma_data
                # Start computation of the next batch of images
                if b < (batch_per_epoch - 1):
                    inds_batch_next = valid_index[inds[(b+1)*self.batch_size:(b+2)*self.batch_size]]
                    params = get_batch_params(inds_batch_next)
                    preprocessing_pool = pool.map_async(_preprocessing_worker, params)

                ## Perform update
                train_err, logpx, klp = self.model._trainer(floatX(xdata),
                                                            floatX(ydata),
                                                            floatX(sigma_data),
                                                            floatX(learning_rate))

                acc_train_err += train_err
                acc_logpx += logpx
                acc_klp += klp
                count += 1
                
            acc_train_err /= count
            acc_logpx /= count
            acc_klp /= count
            print("--- Epoch %d took %f s"%(i, time.time() - start_time))
            print("    Variational Lower Bound: %f, log_px: %f, KL prior: %f"%(acc_train_err,acc_logpx, acc_klp))
        pool.close()


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
