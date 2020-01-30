# -*- coding: utf-8 -*-
"""
This is a file with misc I/0 functions helping to open velocity and image
files.
"""
import os

import numpy as np
import scipy

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

default_float = np.float32
default_suffix_separator = "_"
default_prefix_separator = ""

class AttrDict(dict):
    """New Dictionary which adds the attributes using
    the items as names
    """
    def __getattr__(self, item):
        return self[item]

    def __dir__(self):
        return super().__dir__() + [str(k) for k in self.keys()]

# adding prefix
def add_prefix(name, prefix=None, link=default_prefix_separator):
    """Add prefix to name

    Args:
        name:
        prefix:
        link:

    Returns:
        new name with prefix
    """
    if prefix is None:
        return name
    else:
        return "{0}{1}{2}".format(prefix, link, name)

# remove prefix
def remove_prefix(name, prefix=None, link=default_prefix_separator):
    """Remove prefix to name

    Args:
        name:
        prefix:
        link:

    Returns:
        new name without prefix if it exists
    """
    if prefix is None or not name.startswith(prefix+link):
        return name
    else:
        return name.replace(prefix+link, "")

# adding suffix
def add_suffix(name, suffix=None, link=default_suffix_separator):
    """Add suffix to name

    Args:
        name:
        suffix:
        link:

    Returns:
        new name with suffix
    """
    if suffix is None:
        return name
    else:
        return "{0}{1}{2}".format(name, link, suffix)

# remove suffix
def remove_suffix(name, suffix=None, link=default_suffix_separator):
    """Remove suffix to name

    Args:
        name:
        suffix:
        link:

    Returns:
        new name without suffix
    """
    if suffix is None or not name.endswith(link+suffix):
        return name
    else:
        return name.replace(link+suffix, "")

# Add suffix for error attributes
def add_err_prefix(name, link=default_prefix_separator):
    """Add error (e) prefix to name

    Args
        name: str

    Returns
        name with error prefix
    """
    return add_prefix(name, "e", link=link)

#========================================
# Reading the Circular Velocity from file
#========================================
def read_vc_file(filename, Vcfile_type="ROTCUR"):
    """Read a circular velocity ascii file. File can be of
    type ROTCUR (comments are '!') or ASCII (comments are '#')

    Parameters
    ----------
    filename: str
        name of the file.
    Vcfile_type: str ['ROTCUR']
        'ROTCUR' or 'ASCII'.

    Returns
    -------
    status: int
        0 if all fine. -1 if opening error, -2 if file type not
        recognised
    radius: float array
        Radius sample
    Vc: float array
        Circular velocity as read for radius
    eVc: float array
        Uncertainty on Circular velocity
    """

    dic_comments = {"ROTCUR": "!", "ASCII": "#"}

    # Setting up a few values to 0
    radius = Vc = eVc = 0.

    # Testing the existence of the file
    if not os.path.isfile(filename):
        print('OPENING ERROR: File {0} not found'.format(filename))
        status = -1
    else:
        if Vcfile_type.upper() not in dic_comments.keys():
            print("ERROR: Vc file type not recognised")
            status = -2
        else:
            # Reading the file using the default comments
            Vcdata = np.loadtxt(filename,
                                comments=dic_comments[Vcfile_type.upper()]).T

            # now depending on file type - ROTCUR
            if Vcfile_type.upper() == "ROTCUR":
                selV = (Vcdata[7] == 0) & (Vcdata[6] == 0)
                radius = Vcdata[0][selV]
                Vc = Vcdata[4][selV]
                eVc = Vcdata[5][selV]

            # now - ASCII
            elif Vcfile_type.upper() == "ASCII":
                radius = Vcdata[0]
                Vc = Vcdata[1]
                eVc = np.zeros_like(Vc)

            status = 0

    return status, radius, Vc, eVc

#============================================================
# ----- Extracting the header and data array ------------------
#============================================================
def extract_frame(fits_name, pixelsize=1., verbose=True):
    """Extract 2D data array from fits
    and return the data and the header

    Parameters
    ----------
    fits_name: str
        Name of fits image
    pixelsize:  float
        Will read CDELT1 if it exists. Only used if CDELT does not
        exist.
    verbose:    bool
        Default is True

    Returns
    -------
    data:       float array
        data array from the input image. None if image does not exists.
    h:          header
        Fits header from the input image. None if image does not exists.
    steparc: float
        Step in arcseconds
    """
    if (fits_name is None) or (not os.path.isfile(fits_name)):
        print(('Filename {0} does not exist, sorry!'.format(fits_name)))
        return None, None, 1.0

    else:
        if verbose:
            print(("Opening the Input image: {0}".format(fits_name)))
        # --------Reading of fits-file for grav. pot----------
        data, h = pyfits.getdata(fits_name, header=True)
        # -------------- Fits Header IR Image------------
        naxis1, naxis2 = h['NAXIS1'], h['NAXIS2']
        data = np.nan_to_num(data.reshape((naxis2, naxis1)))

        # Checking the step from the Input image (supposed to be in degrees)
        # If it doesn't exist, we set the step to 1. (arcsec)
        desc = 'CDELT1'
        if desc in h:
            steparc = np.fabs(h[desc] * 3600.)  # calculation in arcsec
            if verbose:
                print('Read pixel size ({0}) of Main Image = {1}'.format(h[desc], steparc))
        else:
            steparc = pixelsize  # in arcsec
            if verbose:
                print("Didn't find a CDELT descriptor, use step={0}".format(steparc))
        return data, h, steparc

# --------------------------------------------------
# Functions to help the sampling
# --------------------------------------------------
def guess_step(Xin, Yin, index_range=[0,100], verbose=False) :
    """Guess the step from a 1 or 2D grid
    Using the distance between points for the range of points given by
    index_range

    Parameters:
    -----------
    Xin, Yin: input (float) arrays
    index_range : tuple or array of 2 integers providing the min and max indices = [min, max]
            default is [0,100]
    verbose: default is False

    Returns
    -------
    step : guessed step (float)
    """
    ## Stacking the first 100 points of the grid and determining the distance
    stackXY = np.vstack((Xin[index_range[0]:index_range[1]], Yin[index_range[0]:index_range[1]])).T
    diffXY = distance.cdist(stackXY, stackXY)

    step = np.min(diffXY[diffXY > 0])
    if verbose:
        print("New step will be %s"%(step))

    return step

def get_extent(Xin, Yin) :
    """Return the extent using the min and max of the X and Y arrays

    Return
    ------
    [xmin, xmax, ymin, ymax]
    """
    return [Xin.min(), Xin.max(), Yin.min(), Yin.max()]

def get_1d_radial_sampling(rmap, nbins):
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

# ============================================================
# -----------Create Radial Profile----------------------------
# ============================================================
def extract_radial_profile(rmap, data, nbins,
                           thetamap=None, verbose=True, angle_wedge=0.0,
                           wedge_size=0.0):
    """Extract a radial profile from input frame
    Input
    -----
    rmap: float array
        Values of the radius.
    data: float array
        Input data values.
    nbins: int
        Number of bins for the radial profile.
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

    # First deriving the max and cutting it in nbins
    rsamp, stepr = get_1d_radial_sampling(rmap, nbins)
    rdata = np.zeros_like(rsamp)
    if thetamap is None:
        thetamap = np.ones_like(rmap)
        wedge_size = 0.0
    else:
        thetamap -= wedge_angle

    # Filling in the values for y (only if there are some selected pixels)
    for i in range(len(rsamp) - 1):
        # Selecting an annulus between two radii and without a wedge
        sel = np.where((rmap >= rsamp[i]) & (rmap < rsamp[i+1])
                    & (((thetamap > wedge_size)
                       & (thetamap < 180.0 - wedge_size))
                    | ((thetamap > 180.0 + wedge_size)
                       & (thetamap < 360. - wedge_size))))
        if len(sel) > 0:
            rdata[i] = np.mean(data[sel], axis=None)

    # Returning the obtained profile
    return rsamp, rdata
