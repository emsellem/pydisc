# -*- coding: utf-8 -*-
"""
This provides a few useful units for this package using astropy units
"""
import numpy as np
import os

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

from astropy import units as u
from astropy.constants import G as Ggrav_cgs

Ggrav = Ggrav_cgs.to(u.km**2 * u.pc / u.s**2 / u.M_sun)
# Defining our default units
pixel = u.pixel
Lsun = u.Lsun
Msun = u.Msun
Lsunpc2 = Lsun / u.pc**2
Msunpc2 = Msun / u.pc**2
kms = u.km / u.s
kmskpc = u.km / u.s / u.kpc
kms2 = (u.km / u.s)**2
km_pc = u.pc.to(u.km)
s_yr = u.yr.to(u.s)

def get_conversion_factor(unit, newunit, equiv=[], verbose=True):
    """Get the conversion factor for astropy units

    Args:
        unit: astropy unit
        newunit: astropy unit

    Returns:
        conversion_factor, validated unit
    """
    try:
        fac = (1 * unit).to(newunit, equivalencies=equiv).value
        return fac, newunit
    except:
        if verbose:
            print("ERROR: cannot convert these units: from {0} to {1}".format(
                    unit, newunit))
        fac = 1.0
        return fac, unit

