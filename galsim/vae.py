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
import cPickle as pickle
from .generative_model import GenerativeGalaxyModel

from deepgen import ladder, gmm_prior_step, resnet_step, dens_step

from numpy.random import randint, permutation
from lasagne.utils import floatX

def _preprocessing_worker(params):
    """
    Outputs an image after applying random rotation, scaling and noise
    padding
    """
    original_gal, stamp_size, pixel_scale = params
    original_gal = galsim.InterpolatedImage(original_gal, pad_factor=4)
    g = original_gal.rotate(galsim.Angle(-np.random.rand()*2*np.pi, galsim.radians)) 
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
                 batch_size=64,
                 n_bands=1,
                 pixel_scale=0.03,
                 model_params=None):
        super(self.__class__, self).__init__(quantities)
        self.stamp_size = stamp_size
        self.batch_size = batch_size
        self.n_bands = n_bands
        self.pixel_scale = pixel_scale

        # Defines optional arguments for sampling
        self.sample_opt_params["noise_pad_size"] = int
        
        n_hidden = 8

        # Create the architecture of the ladder
        p = gmm_prior_step(n_units=[256, 256],
                    n_hidden=n_hidden,
                    IAF_sizes=[[128,128],[128,128]],
                    n_gaussians=512)

        # First resnet layer, 64x64
        resnet_1 = resnet_step(n_filters=32,
                            latent_dim=8,
                            IAF_sizes=[[32,32],[32,32]],
                            prefilter=True)

        resnet_2 = resnet_step(n_filters=64,
                            latent_dim=32,
                             IAF_sizes=[[32,32],[32,32]],
                            downsample=True)
        # Input at 32x32
        resnet_3 = resnet_step(n_filters=128,
                                latent_dim=64,
                                IAF_sizes=[[64,64],[64,64]],
                                downsample=True)

        # Input at 16x16
        resnet_4 = resnet_step(n_filters=256,
                                latent_dim=128,
                                IAF_sizes=[[128,128],[128,128]],
                                downsample=True)
        # Input at 8x8
        resnet_5 = resnet_step(n_filters=256,
                                latent_dim=128,
                                IAF_sizes=[[128,128],[128,128]],
                                downsample=True)

        # Input at 4x4
        dense_5 = dens_step(n_units=256)

        # Build the ladder
        self.model = ladder(n_bands, stamp_size,
                            len(quantities),
                            [resnet_1, resnet_2, resnet_3, resnet_4,resnet_5, dense_5, p],
                            batch_size=batch_size)

        # Default normalisation variables
        self.x_scaling = 1
        self.y_scaling = 1
        self.y_shift = 0

        # Load pre-trained parameters if provided
        if model_params is not None:
            self.x_scaling , self.y_scaling, self.y_shift, net_params = model_params
            self.model.set_params(net_params)


    def fit(self, real_cat, param_cat, valid_index=None,
            learning_rate=0.001, n_epochs=100, lr_step=30, processes=None,
            vmax=None):
        """
        Train the model

        Parameters
        ----------
        
        real_cat
        
        param_cat
        
        """
        # multiprocessing for online generation of training images
        from multiprocessing import Pool
        pool = Pool(processes=processes)
        
        if valid_index is None:
            valid_index = range(real_cat.nobjects)

        ngal = len(valid_index)

        batch_per_epoch = ngal/self.batch_size

        xdata = np.zeros((self.batch_size, self.n_bands, self.stamp_size, self.stamp_size)).astype('float32')
        ydata = np.zeros((self.batch_size, len(self.quantities))).astype('float32')
        
        # Normalise the input image using the median noise variance
        # to make training easier for the network
        self.x_scaling = np.median(np.sqrt(real_cat.variance[valid_index]))
        
        # Normalise y parameters by removing the mean and dividing by std
        self.y_scaling = np.ones(len(self.quantities))
        self.y_shift   = np.zeros(len(self.quantities))
        for j, q in  enumerate(self.quantities):
            self.y_scaling[j] = np.std(param_cat[q])
            self.y_shift[j] = np.mean(param_cat[q])

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
                    params.append((gal_image, self.stamp_size, self.pixel_scale))
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
                ydata = (ydata - self.y_shift) / self.y_scaling
                
                sigma_data = np.log(real_cat.variance[inds_batch] / self.x_scaling**2).reshape((self.batch_size, 1))
                
                # Get the preprocessed images, fails after a minute
                ims = preprocessing_pool.get(timeout=60)
                xdata[:,0,:,:] = np.stack(ims) 
                if vmax is not None:
                    xdata = np.clip(xdata, -vmax, vmax)
                xdata = xdata / self.x_scaling
                
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


    def sample(self, cat, noise=None,  rng=None, x_interpolant=None, k_interpolant=None,
               pad_factor=4, noise_pad_size=0, gsparams=None):
        """
        Samples images from the model conditionned on the input variables
        
        Parameters
        ----------
        
        cat: recarray
            table with the values to sample from 
        """
        
        # Build numpy array from data
        y = np.zeros((len(cat), len(self.quantities)))
        for j, q in  enumerate(self.quantities):
            y[:,j] = cat[q]
        y = (y - self.y_shift) / self.y_scaling

        # First sample code conditioned on y
        z = self.model.sample_prior(y)
        
        # Then, decode the code and apply scaling used during training
        x = self.model.inverse_transform(z, y) * self.x_scaling
        
        # Now, we build an InterpolatedImage for each of these
        ims = []
        for i in range(len(x)):
            im = galsim.Image(np.ascontiguousarray(x[i,0].astype(np.float64)),
                              scale=self.pixel_scale)
            ims.append(galsim.InterpolatedImage(im,
                                                x_interpolant=x_interpolant,           
                                                k_interpolant=k_interpolant,
                                                pad_factor=pad_factor,
                                                noise_pad_size=noise_pad_size,
                                                noise_pad=noise,
                                                rng=rng,
                                                gsparams=gsparams))
        return ims


    def write(self, file_name):
        """
        Exports the trainned model
        """
        model_params = (self.x_scaling , self.y_scaling, self.y_shift, self.model.get_params())
        all_params = [self.stamp_size,
                 self.quantities,
                 self.batch_size,
                 self.n_bands,
                 self.pixel_scale,
                 model_params]
        
        f = file(file_name, 'wb')
        pickle.dump(all_params, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
