# -*- coding: utf-8 -*-
"""
A set of useful functions to derive the gravitational potential
and to help with deprojection
"""

# External packages
import numpy as np

# Default cosmology
from astropy.cosmology import WMAP9

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

#==========================
#  Sech2 Function
#==========================
def sech2(z):
    """Sech2 function using numpy.cosh

    Input
    -----
    z: float

    Returns
    -------
    float - Sech2(z)
    """
    return 1. / (np.cosh(z))**2

#===========================
# Get the proper scale
#===========================
def get_pc_per_arcsec(distance, cosmo=WMAP9):
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
    return kpc_per_arcmin.to(u.pc / u.arcsec)


def clipping_to_rectangle(yin, PA, extent=[0,1,0,1]):
    """Find out the portion of the line within the rectangle
    defined by extent

    Args
        yin: float
            position in Y
        PA: float
            position angle
    """
    PA_rad = np.deg2rad(PA)
    cPA = np.cos(PA_rad)
    sPA = np.sin(PA_rad)

    # Now derive the 4 intersections points
    # and remove the ones which are outside
    x0, x1 = extent[0], extent[1]
    y0, y1 = extent[2], extent[3]

    Xout = np.zeros(4)
    Yout = np.zeros(4)
    Xout[0], Xout[1] = x0, x1
    Yout[2], Yout[3] = y0, y1
    if PA != 0.:
        Yout[0] = (-x0 * cPA - yin) / sPA
        Yout[1] = (-x1 * cPA - yin) / sPA
    else:
        Yout[0] = np.Inf
        Yout[1] = np.Inf

    if cPA != 0.:
        Xout[2] = (- y0 * sPA - yin) / cPA
        Xout[3] = (- y1 * sPA - yin) / cPA
    else:
        Xout[2] = Xout[3] = np.Inf

    list_points = []
    for i in range(4):
        if (Xout[i] >= x0) and (Xout[i] <= x1) \
                    and (Yout[i] >= y0) and (Yout[i] <= y1):
            list_points.append(np.array([Xout[i], Yout[i]]))

    result = np.zeros((2,2))
    for i in range(2):
        result[i, 0] = list_points[i][0]
        result[i, 1] = list_points[i][1]
    return result

# Python program to implement Cohen Sutherland algorithm for line clipping.
# Implementing Cohen-Sutherland algorithm
# Clipping a line from P1 = (x1, y1) to P2 = (x2, y2)
def cohenSutherlandClip(x1, y1, x2, y2, extent=[0,1,0,1]):
    """Takes 2 points (defining a line) and find out where
    it stands within the rectangle defined by extent
    """

    x_min, x_max = extent[0], extent[1]
    y_min, y_max = extent[0], extent[1]
    # Compute region codes for P1, P2
    code1 = computeCode(x1, y1)
    code2 = computeCode(x2, y2)
    accept = False

    def computeCode(x, y):
        """Code for Clipping
        """
        # Defining region codes
        INSIDE = 0  # 0000
        LEFT = 1  # 0001
        RIGHT = 2  # 0010
        BOTTOM = 4  # 0100
        TOP = 8  # 1000
        code = INSIDE
        if x < x_min:  # to the left of rectangle
            code |= LEFT
        elif x > x_max:  # to the right of rectangle
            code |= RIGHT
        if y < y_min:  # below the rectangle
            code |= BOTTOM
        elif y > y_max:  # above the rectangle
            code |= TOP

        return code

    while True:
        # If both endpoints lie within rectangle
        if code1 == 0 and code2 == 0:
            accept = True
            break

        # If both endpoints are outside rectangle
        elif (code1 & code2) != 0:
            break

        # Some segment lies within the rectangle
        else:
            # Line Needs clipping
            # At least one of the points is outside,
            # select it
            x = 1.0
            y = 1.0
            if code1 != 0:
                code_out = code1
            else:
                code_out = code2

            # Find intersection point
            # using formulas y = y1 + slope * (x - x1),
            # x = x1 + (1 / slope) * (y - y1)
            if code_out & TOP:
                # point is above the clip rectangle
                x = x1 + (x2 - x1) * \
                    (y_max - y1) / (y2 - y1)
                y = y_max

            elif code_out & BOTTOM:
                # point is below the clip rectangle
                x = x1 + (x2 - x1) * \
                    (y_min - y1) / (y2 - y1)
                y = y_min

            elif code_out & RIGHT:
                # point is to the right of the clip rectangle
                y = y1 + (y2 - y1) * \
                    (x_max - x1) / (x2 - x1)
                x = x_max

            elif code_out & LEFT:
                # point is to the left of the clip rectangle
                y = y1 + (y2 - y1) * \
                    (x_min - x1) / (x2 - x1)
                x = x_min

            # Now intersection point x,y is found
            # We replace point outside clipping rectangle
            # by intersection point
            if code_out == code1:
                x1 = x
                y1 = y
                code1 = computeCode(x1, y1)

            else:
                x2 = x
                y2 = y
                code2 = computeCode(x2, y2)

    if accept:
        return (x1, y1, x2, y2)
    else:
        return None

