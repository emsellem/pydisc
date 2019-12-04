# -*- coding: utf-8 -*-
"""
This is a file with misc I/0 functions helping to open velocity and image
files.
"""
import numpy as np
import os

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

#================================
# Reading the Circular Velocity
# Data from file
#================================
def read_vcirc_file(filename, Vcfile_type="ROTCUR", finestepR=1.0):
    """Read a circular velocity ascii file. File can be of
    type ROTCUR (comments are '!') or ASCII (comments are '#')

    Parameters
    ----------
    filename: str
        name of the file.
    Vcfile_type: str ['ROTCUR']
        'ROTCUR' or 'ASCII'.
    finestepR: float [1.0]
        Step in radius to interpolate profile

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
    rfine: float array
        Range of radii with finestepR
    Vcfine: float array:
        Interpolated circular velocity in rfine
    """

    dic_comments = {"ROTCUR": "!", "ASCII": "#"}

    # Setting up a few values to 0
    radius = Vc = eVc = rfine = Vcfine = 0.

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
                Vc = Vcdata[1][selV]
                eVc = np.zeros_like(Vc)

            # --- New radius range with fine step in R
            rmax = np.max(radius, axis=None)
            rfine = np.arange(0., rmax, finestepR)

            # Spline interpolation for Vc
            coeff_spline = scipy.interpolate.splrep(radius, Vc, k=1)
            Vcfine = scipy.interpolate.splev(rfine, coeff_spline)
            status = 0

    return status, radius, Vc, eVc, rfine, Vcfine

#============================================================
# ----- Extracting the header and data array ------------------
#============================================================
def extract_frame(image_name, pixelsize=1., verbose=True):
    """Extract 2D data array from fits
    and return the data and the header

    Parameters
    ----------
    image_name: str
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
    if (image_name is None) or (not os.path.isfile(image_name)):
        print(('Filename {0} does not exist, sorry!'.format(image_name)))
        return None, None, 1.0

    else:
        if verbose:
            print(("Opening the Input image: {0}".format(image_name)))
        # --------Reading of fits-file for grav. pot----------
        data = pyfits.getdata(image_name)
        h = pyfits.getheader(image_name)
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
