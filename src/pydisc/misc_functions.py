# -*- coding: utf-8 -*-
"""
A set of useful functions to derive the gravitational potential
and to help with deprojection
"""

# External packages
import numpy as np


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
