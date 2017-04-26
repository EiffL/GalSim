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

from abc import ABCMeta, abstractmethod

class GenerativeGalaxyModel(object):
    """
    Base class defining a generative galaxy model
    """
    __metaclass__ = ABCMeta

    def __init__(self, quantities=[]):
        """
        Initialisation of the generator, must specify a list of predictors
        and features to predict
        """
        self.quantities = quantities

    @abstractmethod
    def fit(self):
        """
        Fits the generative model on provided data
        """
        pass

    @abstractmethod
    def write(self, filename):
        """
        Exports the parameters of the model
        """
        pass

    @abstractmethod
    def sample(self):
        """
        Samples galaxy images from the model`
        """
        pass
