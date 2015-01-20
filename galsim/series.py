# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
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
"""@file series.py
Definitions for Galsim Series class and subclasses.
"""

import galsim
from scipy.misc import factorial
import numpy as np

# Simplified Least Recently Used replacement cache.
# http://code.activestate.com/recipes/577970-simplified-lru-cache/
class LRU_Cache:
    def __init__(self, user_function, maxsize=1024):
        # Link layout:     [PREV, NEXT, KEY, RESULT]
        self.root = root = [None, None, None, None]
        self.user_function = user_function
        self.cache = cache = {}

        last = root
        for i in range(maxsize):
            key = object()
            cache[key] = last[1] = last = [last, root, key, None]
        root[0] = last

    def __call__(self, *key):
        cache = self.cache
        root = self.root
        link = cache.get(key)
        if link is not None:
            link_prev, link_next, _, result = link
            link_prev[1] = link_next
            link_next[0] = link_prev
            last = root[0]
            last[1] = root[0] = link
            link[0] = last
            link[1] = root
            return result
        result = self.user_function(*key)
        root[2] = key
        root[3] = result
        oldroot = root
        root = self.root = root[1]
        root[2], oldkey = None, root[2]
        root[3], oldvalue = None, root[3]
        del cache[oldkey]
        cache[key] = oldroot
        return result

# This metaclass will create classes pre-populated with a `cache` attribute.  This way each
# subclass of `Series` automatically gets its own LRU_Cache.  Note that you can't simply make
# `cache` an attribute of the parent class `Series`, since that makes a single cache shared by
# all of the subclasses of `Series`.
class _cached(type):
    def __new__(cls, clsname, bases, dct):
        dct['cache'] = object()
        dct['kcache'] = object()
        return super(_cached, cls).__new__(cls, clsname, bases, dct)


class Series(object):
    __metaclass__ = _cached
    # Initialize LRU_Cache to store precomputed images.
    def __init__(self, maxcache=1024):
        def basisImg(key):
            idx = key[0]
            nx, ny, scale = key[1]
            obj = self.getBasisFunc(idx)
            return obj.drawImage(nx=nx, ny=ny, scale=scale).array
        def basisKImg(key):
            idx = key[0]
            nx, ny, scale = key[1]
            obj = self.getBasisFunc(idx)
            img, _ = obj.drawKImage(nx=nx, ny=ny, scale=scale)
            return img.array
        self.cache = LRU_Cache(basisImg, maxsize=maxcache)
        self.kcache = LRU_Cache(basisKImg, maxsize=maxcache)

    def getCoeff(self, index):
        raise NotImplementedError("subclasses of Series must define getCoeff() method")

    def getBasisFunc(self, index):
        raise NotImplementedError("subclasses of Series must define getBasisFunc() method")

    def drawImage(self, nx=None, ny=None, scale=None):
        args = (nx, ny, scale)
        return reduce(lambda x,y:x+y,
                      [self.cache((idx,args))*self.getCoeff(idx) for idx in self.indices])

    def drawKImage(self, nx=None, ny=None, scale=None):
        args = (nx, ny, scale)
        return reduce(lambda x,y:x+y,
                      [self.kcache((idx,args))*self.getCoeff(idx) for idx in self.indices])


class SeriesConvolution(Series):
    # Here the index should be tuple(convolutant_subclasses, convolutant_indices)
    #   where convolutant_subclasses = [A, B, C, ...] and
    #   isinstance(A,B,C,..., (GSObject, Series)) is True
    #   and convolutant_indices = outer(A.indices, B.indices, C.indices, ...)
    def __init__(self, *args, **kwargs):
        # First check for number of arguments != 0
        if len(args) == 0:
            # No arguments. Could initialize with an empty list but draw then segfaults. Raise an
            # exception instead.
            raise ValueError("Must provide at least one GSObject")
        elif len(args) == 1:
            # TODO: allow convolution by GSObjects here too!
            if isinstance(args[0], Series):
                args = [args[0]]
            elif isinstance(args[0], list):
                args = args[0]
            else:
                raise TypeError(
                    "Single input argument must be a Series or a list of Series.")
        # Check kwargs
        self.gsparams = kwargs.pop("gsparams", None)
        self.maxcache = kwargs.pop("maxcache", 1024)
        # Make sure there is nothing left in the dict.
        if kwargs:
            raise TypeError("Got unexpected keyword argument(s): %s"%kwargs.keys())
        self.objlist = []
        for obj in args:
            if isinstance(obj, SeriesConvolution):
                self.objlist.extend([o for o in obj.objlist])
            else:
                self.objlist.append(obj)

        # Make self.indices:
        convolutant_subclasses = tuple(obj.__class__ for obj in self.objlist)
        from itertools import product
        convolutant_indices = tuple(product(*[obj.indices for obj in self.objlist]))
        self.indices = tuple((convolutant_subclasses, i) for i in convolutant_indices)

        super(SeriesConvolution, self).__init__(maxcache=self.maxcache)

    def getCoeff(self, index):
        return reduce(lambda x,y:x*y,
                      [obj.getCoeff(idx) for obj, idx in zip(self.objlist, index[1])])

    def getBasisFunc(self, index):
        return galsim.Convolve([obj.getBasisFunc(idx)
                                for obj, idx in zip(self.objlist, index[1])])


class Spergelet(galsim.GSObject):
    """A basis function in the Taylor series expansion of the Spergel profile.

    @param nu               The Spergel index, nu.
    @param scale_radius     The scale radius of the profile.  Typically given in arcsec.
    @param j                Radial index.
    @param q                Azimuthal index.
    @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                            details. [default: None]
    """

    # Initialization parameters of the object, with type information
    _req_params = { "nu" : float, "scale_radius": float, "j":int, "q":int }
    _takes_rng = False
    _takes_logger = False

    # --- Public Class methods ---
    def __init__(self, nu, scale_radius, j, q, gsparams=None):
        galsim.GSObject.__init__(
            self, galsim._galsim.SBSpergelet(nu, scale_radius, j, q, gsparams=gsparams))

    def getNu(self):
        """Return the Spergel index `nu` for this profile.
        """
        return self.SBProfile.getNu()

    def getScaleRadius(self):
        """Return the scale radius for this Spergel profile.
        """
        return self.SBProfile.getScaleRadius()

    def getJQ(self):
        """Return the jq indices for this Spergelet.
        """
        return self.SBProfile.getJ(), self.SBProfile.getQ()


class SpergelSeries(Series):
    _req_params = { "nu" : float, "jmax" : int }
    _opt_params = { "flux" : float }
    _single_params = [ { "half_light_radius" : float, "scale_radius" : float } ]
    _takes_rng = False
    _takes_logger = False

    def __init__(self, nu, jmax, half_light_radius=None, scale_radius=None,
                 flux=1., gsparams=None):
        self.nu=nu
        self.jmax=jmax
        if half_light_radius is not None:
            prof = galsim.Spergel(nu=nu, half_light_radius=half_light_radius)
            self.scale_radius = prof.getScaleRadius()
        else:
            self.scale_radius=scale_radius
        self.flux=flux
        self.gsparams=gsparams

        self.indices=[]
        for j in xrange(jmax+1):
            for q in xrange(j+1):
                self.indices.append((nu, scale_radius, j, q))
        self.indices = tuple(self.indices)

        #for testing
        self.ellip = 0.0
        self.phi0 = 0.0
        self.delta = 0.0

        super(SpergelSeries, self).__init__()

    def getCoeff(self, index):
        _, _, j, q = index
        ellip, phi0, delta = self.ellip, self.phi0, self.delta
        coeff = 0.0
        print
        print "processing j:{} q:{}".format(j, q)
        for m in range(q, j+1):
            if (m+q)%2 == 1:
                continue
            n = (q+m)/2
            num = delta**(j-m) * (1.-delta)**m * ellip**m
            den = 2**(m-1) * factorial(j-m) * factorial((m-q)/2) * factorial((m+q)/2)
            outstr = "j:{} m:{} n:{} num:{} (j-m)!:{} ((m-q)/2)!:{} ((m+q)/2)!:{}"
            print outstr.format(j, m, n, num, factorial(j-m), factorial((m-q)/2), factorial((m+q)/2))
            if q == 0:
                num *= 0.5
            coeff += num/den
        coeff *= self.flux * np.cos(-2.*q*phi0)
        print "j:{}  q:{}  coeff:{}".format(j, q, coeff)
        return coeff

    def getBasisFunc(self, index):
        _, _, j, q = index
        return Spergelet(nu=self.nu, scale_radius=self.scale_radius,
                         j=j, q=q, gsparams=self.gsparams)
