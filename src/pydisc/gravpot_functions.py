# -*- coding: utf-8 -*-
"""
This provides a set of functions associated with a potential
"""
import numpy as np
from scipy import stats

from .misc_io import guess_stepx, get_1d_radial_sampling
from .misc_functions import sech, sech2
from .transform import xy_to_polar
from .local_units import Ggrav, s_yr, km_pc

from astropy.convolution import convolve_fft

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

def get_gravpot_kernel(rpc, hz_pc=None, pc_per_pixel=1.0, softening=0.0, function="sech2"):
    """Calculate the kernel for the potential

    Input
    -----
    pixel_scale: float
    softening: float
        Size of softening in pc
    function: str
        Name of function for the vertical profile
        Can be sech or sech2. If not recognised, will use sech2.
    """

    # Deriving the scale height, just using 1/12. of the size of the box
    if hz_pc is None:
        hz_px = int(rpc.shape[0] / 24.)
    else:
        hz_px = hz_pc / pc_per_pixel

    # Grid in z from -hz to +hz with no central point
    zpx = np.arange(0.5 - hz_px, hz_px + 0.5, 1.)
    zpc = zpx * pc_per_pixel

    # Depending on the vertical distribution
    if function == "sech" :
        h = sech(zpx / hz_px)
    else:
        h = sech2(zpx / hz_px)

    # Integrating over the entire range - Normalised integral
    hn = h / np.sum(h, axis=None)
    kernel = hn[np.newaxis,np.newaxis,...] / (np.sqrt(rpc[...,np.newaxis]**2 + softening**2 + zpc**2))

    return np.sum(kernel, axis=2) / np.sum(kernel)

def get_potential(mass, gravpot_kernel):
    """Calculate the gravitational potential from a disc mass map

    Args:
        mass: 2d array
        gravpot_kernel: 2d array (potential kernel)

    Returns:
        Potential from the convolution of the potential_kernel and mass
    """
    # Initialise array with zeroes
    # G in (km/s)^2 * pc / Msun
    # mass is in Msun 
    # kernel is in 1/pc
    return -Ggrav.value * convolve_fft(mass, gravpot_kernel, 
                                       preserve_nan=False,
                                       normalize_kernel=False,
                                       boundary='wrap')

def get_forces(xpc, ypc, gravpot, PAx=-90.0):
    """Calculate the forces from a given potential

    Args:
        xpc (array): x coordinate in parsec
        ypc (array): y coordinate in parsec
        gravpot (array): gravitational potential
        PAx: position angle of the Ox axis

    Returns:
       F_grad, Fx, Fy, Frad, Ftan (arrays):
           gradient, x y radial and tangential
           components of the forces. 
           Units should be in km^2.s^-2.pc^-1
    """
    # Force from the gradient of the potential
    # gravpot in (km/s)^2 hence F_grad in d/pixel
    F_grad = np.gradient(gravpot)
    # Note that F_grad[1] is along-X, and [0] is along-Y

    # Getting the polar coordinates
    Rpc, theta = xy_to_polar(xpc, ypc)
    theta_rad = np.deg2rad(theta)
    stepx_pc = guess_stepx(xpc)
    stepy_pc = guess_stepx(ypc)

    # Force components in X and Y in (km/s)^2 / pc
    dPhiy = F_grad[0] / stepy_pc
    dPhix = F_grad[1] / stepx_pc

    # If PAx is -90. degrees it means that
    PAx_rad = np.deg2rad(PAx - 90.0)
    # Fx and Fy are now in (km/s)**2 / pc 
    Fx = ( np.cos(PAx_rad) * dPhix + np.sin(PAx_rad) * dPhiy)
    Fy = (-np.sin(PAx_rad) * dPhix + np.cos(PAx_rad) * dPhiy)

    # Radial force vector in outward direction
    Frad =  Fx * np.cos(theta_rad) + Fy * np.sin(theta_rad)
    # Tangential force vector in clockwise direction
    Ftan = -Fx * np.sin(theta_rad) + Fy * np.cos(theta_rad)
    return F_grad, Fx, Fy, Frad, Ftan

def get_vrot_from_force(rpc, Frad):
    """Calculate the rotation velocity from the radial forces

    Args:
        rpc:
        Frad:

    Returns:
        vrot
    """
    # If Frad in (km/s)^2 / pc, so that we return km/s
    return np.sqrt(np.abs(rpc * Frad))

def get_torque(xpc, ypc, Fx, Fy):
    return (xpc * Fy - ypc * Fx)

def get_weighted_torque(xpc, ypc, Fx, Fy, weights):
    return get_torque(xpc, ypc, Fx, Fy) * weights

def get_torque_profiles(xpc, ypc, vel, Fx, Fy, weights, n_rbins=100, pc_per_pixel=1.0):
    """Calculation of the gravity torques
    """
    # Weighted Torque is just Deprojected_Gas * (X * Fy - y * Fx)
    # Hence in (km/s)^2 * Msun / pc^2
    torque_w = get_weighted_torque(xpc, ypc, Fx, Fy, weights).ravel()
    goodw = (weights > 0.).ravel()

    # Average over azimuthal angle and normalization
    rpc = (np.sqrt(xpc**2 + ypc**2)).ravel()
    rsamp, stepr = get_1d_radial_sampling(rpc, n_rbins)

    # And now binning with the various weights
    # Torque in (km/s)^2, weighted Torque in (km/s)^2 * Msun / pc^2
    # Weights should be in Msun/pc2
    weights_mean = stats.binned_statistic(rpc[goodw], weights.ravel()[goodw], statistic='mean', bins=rsamp)
    torque_mean = stats.binned_statistic(rpc[goodw], torque_w[goodw], statistic='mean', bins=rsamp)
    torque_mean_w = torque_mean[0] / weights_mean[0]

    # Angular momentum
    r_mean = stats.binned_statistic(rpc[goodw], rpc[goodw], statistic='mean', bins=rsamp)
    vel_mean = stats.binned_statistic(rpc[goodw], vel.ravel()[goodw], statistic='mean', bins=rsamp)
    # In km2 / s
    ang_mom_mean  = r_mean[0] * km_pc * vel_mean[0]

    # Mass inflow/outflow rate
    # Torque_mean is in (km/s)^2 * Msun / pc^2
    # So T_m / (vel_mean * km_pc) in Msun / pc / s
    # so dm in Msun/yr/pc
    dm = torque_mean[0] * 2. * np.pi * s_yr / (vel_mean[0] * km_pc)

    # Mass inflow/outflow integrated over a certain radius R
    # In Msun/yr
    dm_sum = np.cumsum(dm) * stepr

    return r_mean[0], vel_mean[0], torque_mean[0], torque_mean_w, ang_mom_mean, dm, dm_sum
