# -*- coding: utf-8 -*-
"""
A small module to deal with projection/deprojection,
rotations and other geometrical transformations
"""

import numpy as np
from numpy import deg2rad, rad2deg, cos, sin, arctan, tan, pi

from scipy.interpolate import griddata as gdata
from scipy.ndimage.interpolation import rotate, affine_transform

from .misc_io import default_float

def mirror_grid(self, X, Y) :
    """Find the mirrored grid (above/below major-axis)

    Input
    -----
    Xin, Yin: input grid (numpy arrays)

    Returns
    -------
    Xin -Yin
    """
    return X, -Y

# Rotating a set of X,Y
def rotate_vectors(X=None, Y=None, matrix=np.identity(2), ftype=default_float) :
    """Rotation of coordinates using an entry matrix

    Input
    -----
    X, Y: input grid (arrays)
    matrix : transformation matrix (matrix)

    Returns
    -------
    The rotated array
    """
    shape = X.shape
    newX, newY = np.asarray(matrix *
                            np.vstack((X.ravel(), Y.ravel())).astype(
                                ftype))
    return newX.reshape(shape), newY.reshape(shape)

# Setting up the rotation matrix
def set_rotmatrix(angle=0.0) :
    """Rotation matrix given a specified angle

    Parameters:
    -----------
    angle: angle in radian. Default is 0

    Returns
    -------
    rotation_matrix: matrix
    """
    cosa, sina = cos(angle), sin(angle)
    return np.matrix([[cosa, sina],[-sina, cosa]])


# --------------------------------------------------
# Functions to provide reference matrices
# --------------------------------------------------
# Setting up the stretching matrix
def set_stretchmatrix(coefX=1.0, coefY=1.0) :
    """Streching matrix

    Parameters:
    -----------
    coefX,
    coefY : coefficients (float) for the matrix
              [coefX   0
               0   coefY]
    Returns
    -------
    strectching_matrix: matrix
    """
    return np.array([[coefX, 0],[0, coefY]])

def set_reverseXmatrix() :
    """Reverse X axis using set_strechmatrix(-1.0, 1.0)

    Returns
    -------
    reverse_X_matrix: matrix
    """
    return set_stretchmatrix(-1.0, 1.0)

def set_reverseYmatrix() :
    """Reverse Y axis using set_strechmatrix(1.0, -1.0)

    Return
    ------
    reverse_Y_matrix: matrix
    """
    return set_stretchmatrix(1.0, -1.0)

# --------------------------------------------------
# Resampling the data and visualisation
# --------------------------------------------------
def resample_data(Xin, Yin, Zin, newextent=None, newstep=None, fill_value=np.nan, method='linear', verbose=False) :
    """Resample input data from an irregular grid
    First it derives the limits, then guess the step it should use (if not provided)
    and finally resample using griddata (scipy version).

    The function spits out the extent, and the new grid and interpolated values
    """

    # First test consistency
    test, [Xin, Yin, Zin] = _check_allconsistency_sizes([Xin, Yin, Zin])
    if not test :
        if verbose:
            print("Warning: error in resample_data, not all array size are the same")
        return None, None, None, 0

    # Get the step and extent
    if newstep is None :
        newstep = guess_step(Xin, Yin, verbose=verbose)
    if newextent is None :
        newextent = get_extent(Xin, Yin)
    [Xmin, Xmax, Ymin, Ymax] = newextent

    dX, dY = Xmax - Xmin, Ymax - Ymin
    nX, nY = np.int(dX / newstep + 1), np.int(dY / newstep + 1)
    Xnewgrid, Ynewgrid = np.meshgrid(np.linspace(Xmin, Xmax, nX), np.linspace(Ymin, Ymax, nY))
    newZ = gdata(np.vstack((Xin, Yin)).T, Zin, np.vstack((Xnewgrid.ravel(), Ynewgrid.ravel())).T,
            fill_value=fill_value, method=method)
    return newextent, Xnewgrid, Ynewgrid, newZ.reshape(Xnewgrid.shape)

##############################################################
# -----Rotation and Deprojecting routine----------------------
##############################################################
def deproject_frame(data, PA, inclination=90.0):
    """Returns a deprojected frame given a PA and inclination

    Parameters
    ----------
    data: float array
        Numpy array with input data (image)
    PA: float
        Position angle in degrees.
    inclination: float
        Inclination angle in degrees.

    Returns
    -------
    dep_data: float array
        Deprojected image
    """

    # Reading the shape of the disc array
    Ysize, Xsize = data.shape

    # Creating the new set of needed arrays
    disc_dpj = np.zeros((Ysize + 1, Xsize + 1))
    disc_rec = np.zeros_like(disc_dpj)

    # Recentering the disc
    disc_rec[:Ysize, :Xsize] = data[:, :]
    print("Image to deproject has shape: %f, %f" % (Ysize, Xsize))

    # Phi in radians
    phi = np.deg2rad(inclination)
    # Deprojection Matrix
    dpj_matrix = np.array([[1.0 * np.cos(phi), 0.],
                           [0.0, 1.0]])

    # Rotate Disk around theta
    disc_rot = rotate(np.asarray(disc_rec), PA - 90., reshape=False)

    # Deproject Image
    offy = Ysize / 2 - 1. - (Ysize / 2 - 1.) * np.cos(phi)
    disc_dpj_c = affine_transform(disc_rot, dpj_matrix,
                                  offset=(offy, 0))[:Ysize, :Xsize]

    return disc_dpj_c

def deproject_velocity_profile(V, eV=None, inclin=90.):
    """
    Args:
        V: numpy array
        inclin: float [90]
            Inclination angle in degrees

    Returns:
       numpy array
            Deprojected values for the velocities
    """
    if inclin == 0. :
        return np.full(V.shape, np.inf), np.full(V.shape, np.inf)
    if eV is None:
        eV = np.zeros_like(V)
    return V / sin(np.deg2rad(inclin)), eV / sin(np.deg2rad(inclin))