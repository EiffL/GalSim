# Copyright (c) 2012-2016 by the GalSim developers team on GitHub
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
"""@file shear.py
Redefinition of the Shear class at the Python layer.
"""

import galsim
import numpy as np

class Shear(object):
    """A class to represent shears in a variety of ways.

    The python Shear class (galsim.Shear) can be initialized in a variety of ways to represent shape
    distortions.  A shear is an operation that transforms a circle into an ellipse with
    minor-to-major axis ratio b/a, with position angle beta, while conserving the area (see
    below for a discussion of the implications of this choice).  Given the multiple definitions of
    ellipticity, we have multiple definitions of shear:

    reduced shear |g| = (a - b)/(a + b)
    distortion |e| = (a^2 - b^2)/(a^2 + b^2)
    conformal shear eta, with a/b = exp(eta)
    minor-to-major axis ratio q = b/a

    These can be thought of as a magnitude and a real-space position angle beta, or as two
    components, e.g., g1 and g2, with

    g1 = |g| cos(2*beta)
    g2 = |g| sin(2*beta)

    Note: beta is _not_ the phase of a complex valued shear.  Rather, the complex shear is
    g1 + i g2 = g exp(2 i beta).  Likewise for eta or e.  The phase of the complex value is 2 beta.

    The following are all examples of valid calls to initialize a Shear object:

        >>> s = galsim.Shear()                    # empty constructor sets ellipticity/shear to zero
        >>> s = galsim.Shear(g1=0.05, g2=0.05)
        >>> s = galsim.Shear(g1=0.05)             # assumes g2=0
        >>> s = galsim.Shear(e1=0.05, e2=0.05)
        >>> s = galsim.Shear(e2=0.05)             # assumes e1=0
        >>> s = galsim.Shear(eta1=0.07, eta2=-0.1)
        >>> s = galsim.Shear(eta=0.05, beta=45.0*galsim.degrees)
        >>> s = galsim.Shear(g=0.05, beta=0.25*numpy.pi*galsim.radians)
        >>> s = galsim.Shear(e=0.3, beta=30.0*galsim.degrees)
        >>> s = galsim.Shear(q=0.5, beta=0.0*galsim.radians)

    There can be no mixing and matching, e.g., specifying `g1` and `e2`.  It is permissible to only
    specify one of two components, with the other assumed to be zero.  If a magnitude such as `e`,
    `g`, `eta`, or `q` is specified, then `beta` is also required to be specified.  It is possible
    to initialize a Shear with zero reduced shear by specifying no args or kwargs, i.e.
    galsim.Shear().

    Since we have defined a Shear as a transformation that preserves area, this means that it is not
    a precise description of what happens during the process of weak lensing.  The coordinate
    transformation that occurs during the actual weak lensing process is such that if a galaxy is
    sheared by some `(gamma_1, gamma_2)`, and then sheared by `(-gamma_1, -gamma_2)`, it will in the
    end return to its original shape, but will have changed in area due to the magnification, `mu =
    1/((1.-kappa)**2 - (gamma_1**2 + gamma_2**2))`, which is not equal to one for non-zero shear
    even for convergence `kappa=0`.  Application of a Shear using the GSObject.shear() method does
    not include this area change.  To properly incorporate the effective change in area due to
    shear, it is necessary to either (a) define the Shear object, use the shear() method, and
    separately use the magnify() method, or (b) use the lens() method that simultaneously magnifies
    and shears.
    """
    def __init__(self, *args, **kwargs):

        # There is no valid set of >2 keyword arguments, so raise an exception in this case:
        if len(kwargs) > 2:
            raise TypeError(
                "Shear constructor received >2 keyword arguments: %s"%kwargs.keys())

        if len(args) > 1:
            raise TypeError(
                "Shear constructor received >1 non-keyword arguments: %s"%args)

        # If a component of e, g, or eta, then require that the other component is zero if not set,
        # and don't allow specification of mixed pairs like e1 and g2.
        # Also, require a position angle if we didn't get g1/g2, e1/e2, or eta1/eta2

        # Unnamed arg must be a complex shear
        if len(args) == 1:
            self._g = args[0]
            if not isinstance(self._g, complex):
                raise TypeError("Non-keyword argument to Shear must be complex g1 + 1j * g2")

        # Empty constructor means shear == (0,0)
        elif not kwargs:
            self._g = 0j

        # g1,g2
        elif 'g1' in kwargs or 'g2' in kwargs:
            g1 = kwargs.pop('g1', 0.)
            g2 = kwargs.pop('g2', 0.)
            self._g = g1 + 1j * g2
            if abs(self._g) > 1.:
                raise ValueError("Requested shear exceeds 1: %f"%abs(self._g))

        # e1,e2
        elif 'e1' in kwargs or 'e2' in kwargs:
            e1 = kwargs.pop('e1', 0.)
            e2 = kwargs.pop('e2', 0.)
            absesq = e1**2 + e2**2
            if absesq > 1.:
                raise ValueError("Requested distortion exceeds 1: %s"%np.sqrt(absesq))
            self._g = (e1 + 1j * e2) * self._e2g(absesq)

        # eta1,eta2
        elif 'eta1' in kwargs or 'eta2' in kwargs:
            eta1 = kwargs.pop('eta1', 0.)
            eta2 = kwargs.pop('eta2', 0.)
            eta = eta1 + 1j * eta2
            abseta = abs(eta)
            self._g = eta * self._eta2g(abseta)

        # g,beta
        elif 'g' in kwargs:
            if 'beta' not in kwargs:
                raise TypeError(
                    "Shear constructor requires position angle when g is specified!")
            beta = kwargs.pop('beta')
            if not isinstance(beta, galsim.Angle):
                raise TypeError(
                    "The position angle that was supplied is not an Angle instance!")
            g = kwargs.pop('g')
            if g > 1 or g < 0:
                raise ValueError("Requested |shear| is outside [0,1]: %f"%g)
            self._g = g * np.exp(2j * beta.rad())

        # e,beta
        elif 'e' in kwargs:
            if 'beta' not in kwargs:
                raise TypeError(
                    "Shear constructor requires position angle when e is specified!")
            beta = kwargs.pop('beta')
            if not isinstance(beta, galsim.Angle):
                raise TypeError(
                    "The position angle that was supplied is not an Angle instance!")
            e = kwargs.pop('e')
            if e > 1 or e < 0:
                raise ValueError("Requested distortion is outside [0,1]: %f"%e)
            self._g = self._e2g(e**2) * e * np.exp(2j * beta.rad())

        # eta,beta
        elif 'eta' in kwargs:
            if 'beta' not in kwargs:
                raise TypeError(
                    "Shear constructor requires position angle when eta is specified!")
            beta = kwargs.pop('beta')
            if not isinstance(beta, galsim.Angle):
                raise TypeError(
                    "The position angle that was supplied is not an Angle instance!")
            eta = kwargs.pop('eta')
            if eta < 0:
                raise ValueError("Requested eta is below 0: %f"%eta)
            self._g = self._eta2g(eta) * eta * np.exp(2j * beta.rad())

        # q,beta
        elif 'q' in kwargs:
            if 'beta' not in kwargs:
                raise TypeError(
                    "Shear constructor requires position angle when q is specified!")
            beta = kwargs.pop('beta')
            if not isinstance(beta, galsim.Angle):
                raise TypeError(
                    "The position angle that was supplied is not an Angle instance!")
            q = kwargs.pop('q')
            if q <= 0 or q > 1:
                raise ValueError("Cannot use requested axis ratio of %f!"%q)
            eta = -np.log(q)
            self._g = self._eta2g(eta) * eta * np.exp(2j * beta.rad())

        elif 'beta' in kwargs:
            raise TypeError("beta provided to Shear constructor, but not g/e/eta/q")

        # check for the case where there are 1 or 2 kwargs that are not valid ones for
        # initializing a Shear
        if kwargs:
            raise TypeError(
                "Shear constructor got unexpected extra argument(s): %s"%kwargs.keys())

    # define all the methods to get shear values
    def getG1(self):
        """Return the g1 component of the reduced shear.
        Note: s.getG1() is equivalent to s.g1
        """
        return self._g.real

    def getG2(self):
        """Return the g2 component of the reduced shear.
        Note: s.getG2() is equivalent to s.g2
        """
        return self._g.imag

    def getG(self):
        """Return the magnitude of the reduced shear |g1 + i g2| = sqrt(g1**2 + g2**2)
        Note: s.getG() is equivalent to s.g
        """
        return abs(self._g)

    def getBeta(self):
        """Return the position angle of the reduced shear g exp(2i beta) == g1 + i g2
        Note: s.getBeta() is equivalent to s.beta
        """
        return 0.5 * np.angle(self._g) * galsim.radians

    def getShear(self):
        """Return the reduced shear as a complex number g1 + 1j * g2
        Note: s.getShear() is equivalent to s.shear
        """
        return self._g

    def getE1(self):
        """Return the e1 component of the distortion.
        Note: s.getE1() is equivalent to s.e1
        """
        return self._g.real * self._g2e(abs(self._g)**2)

    def getE2(self):
        """Return the e2 component of the distortion.
        Note: s.getE2() is equivalent to s.e2
        """
        return self._g.imag * self._g2e(abs(self._g)**2)

    def getE(self):
        """Return the magnitude of the distortion |e1 + i e2| = sqrt(e1**2 + e2**2)
        Note: s.getE() is equivalent to s.e
        """
        return abs(self._g) * self._g2e(abs(self._g)**2)

    def getESq(self):
        """Return the magnitude squared of the distortion |e1 + i e2|**2 = e1**2 + e2**2
        Note: s.getESq() is equivalent to s.esq
        """
        return self.e**2

    def getEta(self):
        """Return the magnitude of the conformal shear eta = atanh(e) = 2 atanh(g)
        Note: s.getEta() is equivalent to s.eta
        """
        return abs(self._g) * self._g2eta(abs(self._g))

    # make it possible to access g, e, etc. of some Shear object called name using name.g, name.e
    g1 = property(getG1)
    g2 = property(getG2)
    g = property(getG)
    beta = property(getBeta)

    shear = property(getShear)

    e1 = property(getE1)
    e2 = property(getE2)
    e = property(getE)
    esq = property(getESq)
    eta = property(getEta)

    # Helpers to convert between different conventions
    # Note: These return the scale factor by which to multiply.  Not the final value.
    def _g2e(self, absgsq):
        return 2. / (1.+absgsq)

    def _e2g(self, absesq):
        if absesq > 1.e-4:
            #return (1. - np.sqrt(1.-absesq)) / absesq
            return 1. / (1. + np.sqrt(1.-absesq))
        else:
            # Avoid numerical issues near e=0 using Taylor expansion
            return 0.5 + absesq*(0.125 + absesq*(0.0625 + absesq*0.0390625))

    def _g2eta(self, absg):
        if absg > 1.e-4:
            return 2.*np.arctanh(absg)/absg
        else:
            # This doesn't have as much trouble with accuracy, but have to avoid absg=0,
            # so might as well Taylor expand for small values.
            absgsq = absg * absg
            return 2. + absgsq*((2./3.) + absgsq*0.4)

    def _eta2g(self, abseta):
        if abseta > 1.e-4:
            return np.tanh(0.5*abseta)/abseta
        else:
            absetasq = abseta * abseta
            return 0.5 + absetasq*((-1./24.) + absetasq*(1./240.))

    # define all the various operators on Shear objects
    def __neg__(self): return Shear(-self._g)

    # order of operations: shear by other._shear, then by self._shear
    def __add__(self, other):
        return Shear((self._g + other._g) / (1. + self._g.conjugate() * other._g))

    # order of operations: shear by -other._shear, then by self._shear
    def __sub__(self, other): return self + (-other)

    def __eq__(self, other): return self._g == other._g
    def __ne__(self, other): return not self.__eq__(other)

    def getMatrix(self):
        """Return the matrix that tells how this shear acts on a position vector:

        If a field is sheared by some shear, s, then the position (x,y) -> (x',y')
        according to:

        [ x' ] = S [ x ]
        [ y' ]     [ y ]

        where S = s.getMatrix().

        Specifically, the matrix is S = (1-g^2)^(-1/2) [ 1+g1 ,  g2  ]
                                                       [  g2  , 1-g1 ]
        """
        return np.array([[ 1.+self.g1,  self.g2   ],
                         [  self.g2  , 1.-self.g1 ]]) / np.sqrt(1.-self.g**2)

    def rotationWith(self, other):
        """Return the rotation angle associated with the addition of two shears.

        The effect of two shears is not just a single net shear.  There is also a rotation
        associated with it.  This is easiest to understand in terms of the matrix representations:

        If s3 = s1 + s2, and the corresponding shear matrices are S1,S2,S3, then S3 R = S1 S2,
        where R is a rotation matrix:

        R = [ cos(theta) , -sin(theta) ]
            [ sin(theta) ,  cos(theta) ]

        and theta is the return value (as a galsim.Angle) from s1.rotationWith(s2).
        """
        # Save a little time by only working on the first column.
        S3 = self.getMatrix().dot(other.getMatrix()[:,:1])
        R = (-(self + other)).getMatrix().dot(S3)
        theta = np.arctan2(R[1,0], R[0,0])
        return theta * galsim.radians

    def __repr__(self):
        return 'galsim.Shear(%r)'%(self.shear)

    def __str__(self):
        return 'galsim.Shear(g1=%s,g2=%s)'%(self.g1,self.g2)

    def __hash__(self): return hash(self._g)
