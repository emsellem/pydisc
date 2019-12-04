#!/usr/bin/python

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
import pylab as p  # Standard Libraries
import numpy as np
from scipy.ndimage.interpolation import rotate, affine_transform


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


# ============================================================
##############################################################
# -----Rotation and Deprojecting routine----------------------
##############################################################
def deproject_frame(data, PA, inclination=90.0, plot=False):
    """Returns a deprojected frame given a PA and inclination

    Parameters
    ----------
    data: float array
        Numpy array with input data (image)
    PA: float
        Position angle in degrees.
    inclination: float
        Inclination angle in degrees.
    plot: bool
        Default is False. If True, will create a matplotlib
        (simple) figure.

    Returns
    -------
    dep_data: float array
        Deprojected image
    """

    # Reading the shape of the disk array
    Ysize, Xsize = data.shape

    # Creating the new set of needed arrays
    disk_dpj = np.zeros((Ysize + 1, Xsize + 1))
    disk_rec = np.zeros_like(disk_dpj)

    # Recentering the disk
    disk_rec[:Ysize, :Xsize] = data[:, :]
    print("Image to deproject has shape: %f, %f" % (Ysize, Xsize))

    # If plot, produce a matplotlib figure
    if plot:
        p.figure(1)
        p.clf()
        p.imshow(disk_rec)
        p.title("Transfer of the rectified Image before deprojection")
        p.draw()
        if stop_program(): sys.exit(0)

    # Phi in radians
    phi = np.deg2rad(inclination)
    # Deprojection Matrix
    dpj_matrix = np.array([[1.0 * np.cos(phi), 0.],
                           [0.0, 1.0]])

    # Rotate Disk around theta
    disk_rot = rotate(np.asarray(disk_rec), PA - 90., reshape=False)

    # Deproject Image
    offy = Ysize / 2 - 1. - (Ysize / 2 - 1.) * np.cos(phi)
    disk_dpj_c = affine_transform(disk_rot, dpj_matrix,
                                  offset=(offy, 0))[:Ysize, :Xsize]

    return disk_dpj_c


# ============================================================
# -----------Create Radial Profile----------------------------
# ============================================================
def extract_profile(rmap, data, nbins, verbose=True):
    """Extract a radial profile from input frame
    Input
    -----
    rmap: float array
        Values of the radius.
    data: float array
        Input data values.
    nbins: int
        Number of bins for the radial profile.
    verbose: bool
        Default is True (print information)

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

    # First deriving the max and cutting it in nbins
    rsamp, stepr = get_rsamp(rmap, nbins)
    rdata = np.zeros_like(rsamp)

    # Filling in the values for y (only if there are some selected pixels)
    for i in range(len(rsamp) - 1):
        sel = np.where((rmap >= rsamp[i]) & (rmap < rsamp[i + 1]))  ##== selecting an annulus between two bins
        if len(sel) > 0:
            rdata[i] = np.mean(data[sel], axis=None)

    # Returning the obtained profile
    return rsamp, rdata


def get_rsamp(rmap, nbins):
    """Get radius values from a radius map
    Useful for radial profiles

    Parameters
    ----------
    rmap: 2D array
        Input map
    nbins: int
        Number of bins for the output

    Returns
    -------
    rsamp: 1D array
        Array of radii for this map
    rstep: float
        Radial step
    """
    # First deriving the max and cutting it in nbins
    maxr = np.max(rmap, axis=None)

    # Adding 1/2 step
    stepr = maxr / (nbins * 2)
    rsamp = np.linspace(0., maxr + stepr, nbins)
    if nbins > 1:
        rstep = rsamp[1] - rsamp[0]
    else:
        rstep = 1.0

    return rsamp, rstep
