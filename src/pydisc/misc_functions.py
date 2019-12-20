# -*- coding: utf-8 -*-
"""
A set of useful functions to derive the gravitational potential
and to help with deprojection
"""

# External packages
import numpy as np
from scipy.odr import Model, ODR, RealData

#==========================
#  Test to stop program
#==========================
def stop_program():
    """Small function to ask for input and stop if needed
    """
    ok = input("Press S to Stop, and any other key to continue...\n")
    if ok in ["S", "s"]:
        return True
    return False


#==========================
#  Sech Function
#==========================
def sech(z):
    """Sech function using numpy.cosh

    Input
    -----
    z: float

    Returns
    -------
    float - Sech(z)
    """
    return 1. / np.cosh(z)

#===========================
# Get the proper scale
#===========================
def get_pc_per_arcsec(distance, cosmo=None):
    """

    Args:
        distance: float
            Distance in Mpc
        cosmo: astropy.cosmology
            Cosmology. Default is None: will then
            use the default_cosmology from astropy

    Returns:
        pc_per_arcsec: float
            Conversion parsec per arcsecond

    """
    from astropy.cosmology import default_cosmology
    from astropy.coordinates import Distance
    from astropy import units as u

    # Use default cosmology from astropy
    if cosmo is None:
        cosmo = default_cosmology.get()

    # Use astropy units
    dist = Distance(distance, u.Mpc)
    # get the corresponding redshift
    redshift = dist.compute_z(cosmo)
    # And nore the proper conversion
    kpc_per_arcmin = cosmo.kpc_proper_per_arcmin(redshift)
    return kpc_per_arcmin * 1000. / 60.

def linear_fit(ab, x):
    """Linear function of x
    Args:
        ab: [a, b]
            Constant and slope
        x: array or float
            Input data

    Returns:
        a + b * x
    """
    return ab[0] + ab[1] * x

def fit_slope(X, Y, eX=None, eY=None):
    """Will fit a Y vs X with errors using the scipy
    ODR functionalities and a linear fitting function

    Args:
        X:
        Y: input arrays
        eX:
        eY: input uncertaintites [None]

    Returns:
        odr output
    """

    # First filtering the Nan
    good = ~np.isnan(X)
    if eX is None: eX_good = None
    else: eX_good = eX[good]
    if eY is None: eY_good = None
    else: eY_good = eY[good]

    linear = Model(linear_fit)
    odr_data = RealData(X[good], Y[good], sx=eX_good, sy=eY_good)
    odr_run = ODR(odr_data, linear, beta0=[2, 0])
    return odr_run.run()