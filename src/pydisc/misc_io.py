# -*- coding: utf-8 -*-
"""
This is a file with misc I/0 functions helping to open velocity and image
files.
"""
import os

import numpy as np
from scipy.spatial import distance
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

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
        if item in self.keys():
            return self[item]
        else:
            raise AttributeError("'AttrDict' object has no attribute '%s'" % item)

    def __dir__(self):
        return super(AttrDict).__dir__() + [str(k) for k in self.keys()]


# adding prefix
def add_prefix(name, prefix=None, separator=default_prefix_separator):
    """Add prefix to name, except if prefix is None or an empty string ''

    Args:
        name: str
        prefix: str [None]
        separator: str

    Returns:
        new name with prefix
    """
    if prefix is None or prefix=="":
        return name
    else:
        return "{0}{1}{2}".format(prefix, separator, name)

# remove prefix
def remove_prefix(name, prefix=None, separator=default_prefix_separator):
    """Remove prefix to name. If the link separator is present
    it will be removed too.

    Args:
        name: str
        prefix: str [None]
        separator: str

    Returns:
        new name without prefix if it exists
    """
    # If none or wrong start just return the name
    if prefix is None or not name.startswith(prefix):
        return name
    # if it starts with the right prefix+separator
    elif name.startswith(prefix+separator):
        return name.replace(prefix+separator, "")
    # If only the prefix is present, remove it
    elif name.startswith(prefix):
        return name.replace(prefix, "")

# adding suffix
def add_suffix(name, suffix=None, separator=default_suffix_separator):
    """Add suffix to name, except if it is None or an empty str ''

    Args:
        name: str
        suffix: str
        separator: str

    Returns:
        new name with suffix
    """
    if suffix is None or suffix=="" or name is None:
        return name
    else:
        return "{0}{1}{2}".format(name, separator, suffix)

# remove suffix
def remove_suffix(name, suffix=None, separator=default_suffix_separator):
    """Remove suffix to name

    Args:
        name: str
        suffix: str [None]
        separator: str

    Returns:
        new name without suffix
    """
    if suffix is None or not name.endswith(suffix):
        return name
    elif name.endswith(suffix+separator):
        return name.replace(separator+suffix, "")
    elif name.endswith(suffix):
        return name.replace(suffix, "")

# Add suffix for error attributes
def add_err_prefix(name, separator=default_prefix_separator):
    """Add error (e) prefix to name

    Args
        name: str
        separator: str

    Returns
        name with error prefix
    """
    return add_prefix(name, "e", separator=separator)

#========================================
# Reading the Circular Velocity from file
#========================================
def read_vc_file(filename, filetype="ROTCUR"):
    """Read a circular velocity ascii file. File can be of
    type ROTCUR (comments are '!') or ASCII (comments are '#')

    Parameters
    ----------
    filename: str
        name of the file.
    filetype: str ['ROTCUR']
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
        if filetype.upper() not in dic_comments.keys():
            print("ERROR: Vc file type not recognised")
            status = -2
        else:
            # Reading the file using the default comments
            Vcdata = np.loadtxt(filename,
                                comments=dic_comments[filetype.upper()]).T

            # now depending on file type - ROTCUR
            if filetype.upper() == "ROTCUR":
                selV = (Vcdata[7] == 0) & (Vcdata[6] == 0)
                radius = Vcdata[0][selV]
                Vc = Vcdata[4][selV]
                eVc = Vcdata[5][selV]

            # now - ASCII
            elif filetype.upper() == "ASCII":
                radius = Vcdata[0]
                Vc = Vcdata[1]
                eVc = np.zeros_like(Vc)

            status = 0

    return status, radius, Vc, eVc

#============================================================
# ----- Extracting the header and data array ------------------
#============================================================
def extract_fits(fits_name, pixelsize=1., verbose=True):
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

        try:
            thiswcs = WCS(fits_name)
            pixel_scales = proj_plane_pixel_scales(thiswcs)
            steparc = np.fabs(pixel_scales * 3600.)[:2]
            if verbose:
                print('Read pixel size of Main Image = {}'.format(steparc))
        except:
            steparc = pixelsize  # in arcsec
            if verbose:
                print("Didn't find a WCS: will use default step={}".format(steparc))

        return data, h, steparc

# --------------------------------------------------
# Functions to help the sampling
# --------------------------------------------------

def guess_stepx(Xin):
    pot_step = np.array([np.min(np.abs(np.diff(Xin, axis=i))) for i in range(Xin.ndim)])
    return np.min(pot_step[pot_step > 0.])

def guess_stepxy(Xin, Yin, index_range=[0,100], verbose=False) :
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
    # Stacking the first 100 points of the grid and determining the distance
    stackXY = np.vstack((Xin.ravel()[index_range[0]: index_range[1]], Yin.ravel()[index_range[0]: index_range[1]]))
#    xybest = kdtree.KDTree(stackXY).query(stackXY)
#    step = np.linalg.norm(xybest[1] - xybest[0])
    diffXY = np.unique(distance.pdist(stackXY.T))
    step = np.min(diffXY[diffXY > 0])
    if verbose:
        print("New step will be %s"%(step))

    return step

def cover_linspace(start, end, step):
    # First compute how many steps we have
    npix_f = (end - start) / step
    # Then take the integer part
    npix = np.int(np.ceil(npix_f))
    split2 = (npix * step - (end - start)) / 2.
    # Residual split on the two sides
    return np.linspace(start - split2, end + split2, npix+1)

def get_extent(Xin, Yin) :
    """Return the extent using the min and max of the X and Y arrays

    Return
    ------
    [xmin, xmax, ymin, ymax]
    """
    return [np.min(Xin), np.max(Xin), np.min(Yin), np.max(Yin)]

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

