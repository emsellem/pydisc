# -*- coding: utf-8 -*-
"""
A small module to deal with projection/deprojection,
rotations and other geometrical transformations
"""

import numpy as np
from numpy import cos, sin, pi

from scipy.interpolate import griddata as gdata
from scipy.ndimage.interpolation import rotate, affine_transform
from scipy import stats

from .misc_io import default_float, guess_stepxy, \
                     get_extent, get_1d_radial_sampling
from .check import _check_consistency_sizes

def mirror_grid(X, Y):
    """Find the mirrored grid (above/below major-axis)

    Args:
        X (array):
        Y (array): input grid (numpy arrays)

    Returns:
        X, -Y
    """
    return X, -Y

def rotate_vectors(X=None, Y=None, matrix=np.identity(2), ftype=default_float,
                   origin=[0,0]):
    """Rotation of coordinates using an entry matrix

    Args:
        X (array):
        Y (array): input grid
        matrix (array):  transformation matrix
        ftype: type of the array (default is float)
        origin (list of 2 floats): origin location, x, y of the rotation

    Returns:
        The rotated array

    """
    shape = X.shape
    newX, newY = np.asarray(matrix @
                            np.vstack((X.ravel() - origin[0],
                                       Y.ravel() - origin[1])).astype(
                                ftype))
    return (newX+origin[0]).reshape(shape), (newY+origin[1]).reshape(shape)

# Setting up the rotation matrix
def set_rotmatrix(angle=0.0):
    """Rotation matrix given a specified angle

    Parameters:
    -----------
    angle: angle in radian. Default is 0

    Returns
    -------
    rotation_matrix: matrix
    """
    cosa, sina = cos(angle), sin(angle)
    return np.array([[cosa, sina],[-sina, cosa]])

# --------------------------------------------------
# Functions to provide reference matrices
# --------------------------------------------------
# Setting up the stretching matrix
def set_stretchmatrix(coefX=1.0, coefY=1.0):
    """Stretching matrix

    Args:
        coefX:
        coefY:coefficients (float) for the matrix
              [coefX   0
               0   coefY]

    Returns:
    strectching_matrix: matrix
    """
    return np.array([[coefX, 0],[0, coefY]])

def set_reverseXmatrix():
    """Reverse X axis using set_strechmatrix(-1.0, 1.0)

    Returns
    -------
    reverse_X_matrix: matrix
    """
    return set_stretchmatrix(-1.0, 1.0)

def set_reverseYmatrix():
    """Reverse Y axis using set_strechmatrix(1.0, -1.0)

    Return
    ------
    reverse_Y_matrix: matrix
    """
    return set_stretchmatrix(1.0, -1.0)

def regrid_XY(Xin, Yin, newextent=None, newstep=None):
    """Regrid a set of X,Y input from an irregular grid
    First it derives the limits, then guess the
    step it should use (if not provided)

    The function spits out the extent, and the new grid
    """

    # First test consistency
    test = _check_consistency_sizes([Xin, Yin])
    if not test :
        if verbose:
            print("Warning: not all array size are the same")
        return None, None, None

    # Get the step and extent
    if newstep is None :
        newstep = guess_stepxy(Xin, Yin)
    if newextent is None :
        newextent = get_extent(Xin, Yin)
    [Xmin, Xmax, Ymin, Ymax] = newextent

    dX, dY = Xmax - Xmin, Ymax - Ymin
    nX, nY = np.int(dX / newstep + 1), np.int(dY / newstep + 1)
    Xnewgrid, Ynewgrid = np.meshgrid(np.linspace(Xmin, Xmax, nX),
                                     np.linspace(Ymin, Ymax, nY))
    return newextent, newstep, Xnewgrid, Ynewgrid

def rotate_data(Xin, Yin, Zin, angle=0., cx=0., cy=0.):
    rx, ry = rotxyC(Xin, Yin, cx=cx, cy=cy, angle=angle)
    return regrid_XYZ(rx, ry, Zin)

# --------------------------------------------------
# Resampling X, Y and Z (first regrid X, Y)
# --------------------------------------------------
def regrid_XYZ(Xin, Yin, Zin, newextent=None, newstep=None,
                  fill_value=np.nan, method='linear'):
    """Resample input data from an irregular grid
    First it derives the limits, then guess the
    step it should use (if not provided)
    and finally resample using griddata (scipy version).

    The function spits out the extent, and the new
    grid and interpolated values
    """

    if Zin is None: return None
    newextent, newstep, newX, newY = regrid_XY(Xin, Yin, newextent, newstep)
    newZ = regrid_Z(Xin, Yin, Zin, newX, newY,
                    fill_value=fill_value, method=method)
    return newextent, newX, newY, newZ

# --------------------------------------------------
# Resampling Z according to input X, Y
# --------------------------------------------------
def regrid_Z(Xin, Yin, Zin, newX, newY, fill_value=np.nan, method='linear'):
    """Resample input data from an irregular grid
    First it derives the limits, then guess the step
    it should use (if not provided)
    and finally resample using griddata (scipy version).

    The function spits out the extent, and the new
    grid and interpolated values
    """

    if Zin is None: return None
    newZ = gdata(np.vstack((Xin.ravel(), Yin.ravel())).T, Zin.ravel(),
                 np.vstack((newX.ravel(), newY.ravel())).T,
                 fill_value=fill_value, method=method)
    return newZ.reshape(newX.shape)

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
    print("Image to deproject has shape: {0:d}, {1:d}".format(Ysize, Xsize))

    # Creating the new set of needed arrays
    disc_rec = np.zeros_like(data)

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

def deproject_velocities(V, eV=None, inclin=90.):
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

def interpolate_profile(x, data, edata=None, step=1.0):
    """Interpolate from a 1d profile

    Args:
        x:
        data:
        edata:

    Returns:

    """
    # New radii
    xmax = np.max(x, axis=None)
    xmin = np.min(x, axis=None)
    xfine = np.arange(xmin, xmax, step)

    # Spline interpolation for data
    coeff_spline = scipy.interpolate.splrep(x, data, k=1)
    dfine = scipy.interpolate.splev(xfine, coeff_spline)

    # Spline interpolation for edata
    if edata is None:
        edfine = None
    else:
        coeff_espline = scipy.interpolate.splrep(x, edata, k=1)
        edfine = scipy.interpolate.splev(xfine, coeff_espline)
    return xfine, dfine, edfine

def xy_to_polar(x, y, cx=0.0, cy=0.0, angle=0.):
    """
    Convert x and y coordinates into polar coordinates

    cx and cy: Center in X, and Y. 0 by default.
    angle : angle in degrees
         (Counter-clockwise from vertical)
         This allows to take into account some rotation
         and place X along the abscissa
         Default is None and would be then set for no rotation

    Return : R, theta (in degrees)
    """
    # If the angle does not have X along the abscissa, rotate
    if np.mod(angle, 180.) != 0.0 :
        x, y = rotxyC(x, y, cx=cx, cy=cy, angle=angle)
    else :
        x, y = x - cx, y - cy

    # Polar coordinates
    r = np.sqrt(x**2 + y**2)

    # Now computing the true theta
    theta = np.zeros_like(r)
    theta[(x == 0.) & (y >= 0.)] = pi / 2.
    theta[(x == 0.) & (y < 0.)] = -pi / 2.
    theta[(x < 0.)] = np.arctan(y[(x < 0.)] / x[(x < 0.)]) + pi
    theta[(x > 0.)] = np.arctan(y[(x > 0.)] / x[(x > 0.)])
    return r, np.rad2deg(theta)

def polar_to_xy(r, theta):
    """
    Convert x and y coordinates into polar coordinates

    r: float array
    Theta: float array [in Degrees]

    Return :x, y
    """

    theta_rad = np.deg2rad(theta)
    return r * np.cos(theta_rad), r * np.sin(theta_rad)

def rotxC(x, y, cx=0.0, cy=0.0, angle=0.0):
    """ Rotate by an angle (in degrees)
        the x axis with a center cx, cy

        Return rotated(x)
    """
    angle_rad = np.deg2rad(angle)
    return (x - cx) * np.cos(angle_rad) + (y - cy) * np.sin(angle_rad)

def rotyC(x, y, cx=0.0, cy=0.0, angle=0.0):
    """ Rotate by an angle (in degrees)
        the y axis with a center cx, cy

        Return rotated(y)
    """
    angle_rad = np.deg2rad(angle)
    return (cx - x) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad)

def rotxyC(x, y, cx=0.0, cy=0.0, angle=0.0):
    """ Rotate both x, y by an angle (in degrees)
        the x axis with a center cx, cy

        Return rotated(x), rotated(y)
    """
    # First centring
    xt = x - cx
    yt = y - cy

    # Then only rotation
    return rotxC(xt, yt, angle=angle), rotyC(xt, yt, angle=angle)

def extract_radial_profile_fromXY(X, Y, data, nbins=None,
                                  verbose=True,
                                  wedge_size=0.0, wedge_angle=0.):
    """Extract a radial profile from an X,Y, data grod

    Args:
        X:
        Y:
        data:
        nbins:
        verbose:
        wedge_size:
        wedge_angle:

    Returns:
        R, profile
    """

    rmap, thetamap = xy_to_polar(X, Y)
    return extract_radial_profile(rmap, np.nan_to_num(data), nbins=nbins,
                                  thetamap=thetamap,
                                  verbose=verbose, wedge_size=wedge_size,
                                  wedge_angle=wedge_angle)

def extract_radial_profile(rmap, data, nbins=None,
                           thetamap=None, verbose=True,
                           wedge_size=0.0, wedge_angle=0.):
    """Extract a radial profile from input frame
    Given theta and r maps

    Input
    -----
    rmap: float array
        Values of the radius.
    data: float array
        Input data values.
    nbins: int [None]
        Number of bins for the radial profile.
        If None, using an estimate from the input rmap size.
    wedge_angle: float [0]
        Position angle of the wedge to exclude
    wedge_size: float [0]
        Size of the wedge to exclude on each side
    verbose: bool
        Default is True (print information)
    thetamap: 2D array
        Map of theta values (in degrees)

    Returns
    -------
    rsamp: float array
        Radial array (1D)
    rdata: float array
        Radial values (1D)
    """
    # Printing more in case of verbose
    if verbose:
        print("Deriving the radial profile ... \n")

    if nbins is None:
        nbins = np.int(np.sqrt(rmap.size) * 1.5)

    # First deriving the max and cutting it in nbins
    rsamp, stepr = get_1d_radial_sampling(rmap, nbins)
    if thetamap is None:
        thetamap = np.ones_like(rmap)
        wedge_size = 0.0
    else:
        thetamap -= wedge_angle

    # Filling in the values for y (only if there are some selected pixels)
    sel_wedge = (thetamap > wedge_size) & (thetamap < 180.0 - wedge_size)
    rdata, bin_edges, bin_num = stats.binned_statistic(rmap[sel_wedge], data[sel_wedge],
                                                       statistic='mean', bins=rsamp)
    # Returning the obtained profile
    return rsamp[:-1], rdata

