# -*- coding: utf-8 -*-
"""
This is a file with misc I/0 functions helping to open velocity and image
files.
"""
import os

import numpy as np
from scipy import stats
from scipy.spatial import distance, kdtree
from . import transform
from . import local_units as lu

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


def extract_prefixes_from_kwargs(keywords, given_arglist):
    """Extract list of keywords ending with a prefix
    assuming they all end with a string belonging to
    a given list of args

    Attributes
        keywords: list of str
            Keywords to test
        given_arglist: list of str
            Fixed list of args

    Returns
        Dictionary with prefixes as keys and list of found
        args as values
    """
    dict_kwarg = {}
    for arg in given_arglist:
        # We look for the keywords which starts with one of the arg
        found_prefixes = [kwarg.replace(arg, "") for key in keywords if key.endswith(arg)]
        # For all of these, we extract the arg and add it to the dictionary
        for prefix in found_prefixes:
            if prefix in dict_kwarg.keys():
                dict_kwarg[prefix].append(arg)
            else:
                dict_kwarg[prefix] = [arg]

    return dict_kwarg

def extract_suffixes_from_kwargs(keywords, given_arglist, separator=""):
    """Extract list of keywords starting with a suffix
    assuming they all start with a string belonging to
    a given list of args

    Attributes
        keywords: list of str
            Keywords to test
        given_arglist: list of str
            Fixed list of args

    Returns
        Dictionary with suffixes as keys and list of found
        args as values
    """
    dict_kwarg = {}
    # First work out the ambiguous names in the arg list
    dict_doublet = {}
    for arg in given_arglist:
        for arg2 in given_arglist:
            if arg in arg2 and arg != arg2:
                # arg is contained in arg2
                if arg in dict_doublet.keys():
                    dict_doublet[arg].append(arg2)
                else:
                    dict_doublet[arg] = [arg2]

    for arg in given_arglist:
        if arg in dict_doublet.keys():
            larg2 = dict_doublet[arg]
        else:
            larg2 = ["###"]
        # We look for the keywords which starts with one of the arg
        found_suffixes = []
        for key in keywords:
            if key.startswith(arg) and not key.startswith(tuple(larg2)):
                if key == arg:
                    found_suffixes.append(key.replace(arg, ""))
                else:
                    found_suffixes.append(key.replace(arg + separator, ""))
        # For all of these, we extract the arg and add it to the dictionary
        for suffix in found_suffixes:
            if suffix in dict_kwarg.keys():
                dict_kwarg[suffix].append(arg)
            else:
                dict_kwarg[suffix] = [arg]

    return dict_kwarg

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
    if suffix is None or suffix=="":
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

    rmap, thetamap = transform.xy_to_polar(X, Y)
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

