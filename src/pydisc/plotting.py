# -*- coding: utf-8 -*-
"""
A set of useful functions to derive the gravitational potential
and to help with deprojection
"""

__version__ = '1.1.2 (04-12, 2019)'

# Changes --
#   04/12/19- EE - v1.1.2: transferred from pytorque
#   25/06/19- EE - v1.1.0: Python 3
#   13/04/07- EE - v1.0.1: Addition of stop_program()

"""
Import of the required modules
"""
# Other important packages
import matplotlib
from matplotlib import pyplot as pl

def visualise_data(Xin, Yin, Zin, newextent=None, fill_value=np.nan, method='linear', verbose=False, newstep=None, **kwargs) :
    """Visualise a data set via 3 input arrays, Xin, Yin and Zin. The shapes of these arrays should be the
    same.

    Input
    =====
    Xin, Yin, Zin: 3 real arrays
    newextent : requested extent [xmin, xmax, ymin, ymax] for the visualisation. Default is None
    fill_value : Default is numpy.nan
    method : 'linear' as Default for the interpolation, when needed.
    """
    extent, newX, newY, newZ = resample_data(Xin, Yin, Zin, newextent=newextent, newstep=newstep, fill_value=fill_value, method=method,
            verbose=verbose)
    pl.clf()
    pl.imshow(newZ, extent=extent, **kwargs)
    return extent, newX, newY, newZ
