# -*- coding: utf-8 -*-
"""
This provides a set of functions associated with a potential
"""
import numpy as np
from scipy import stats

from .misc_functions import sech, sech2
from .transform import xy_to_polar

from astropy.constants import G as Ggrav
from astropy.convolution import convolve_fft

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"


def get_gravpot_kernel(rpc, hz_pc=None, softening=0.0, function="sech2"):
    """Calculate the kernel for the potential

    Input
    -----
    softening: float
        Size of softening in pc
    function: str
        Name of function for the vertical profile
        Can be sech or sech2. If not recognised, will use sech2.
    """

    # Deriving the scale height, just using 1/12. of the size of the box
    guess_step = np.min(np.abs(np.diff(rpc)))
    if hz_pc is None:
        hz_px = np.int(rpc.shape[0] / 24.)
    else:
        hz_px = hz_pc / guess_step

    # Grid in z from -hz to +hz with no central point
    zpx = np.arange(0.5 - hz_px, hz_px + 0.5, 1.)
    zpc = zpx * guess_step

    # Depending on the vertical distribution
    if function == "sech" :
        h = sech(zpx / hz_px)
    else:
        h = sech2(zpx / hz_px)

    # Integrating over the entire range - Normalised integral
    hn = h / np.sum(h, axis=None)

    kernel = hn[np.newaxis,np.newaxis,...] / (np.sqrt(rpc[...,np.newaxis]**2 + softening**2 + zpc**2))
    return np.sum(kernel, axis=2)

def get_potential(mass, gravpot_kernel):
    """Calculate the gravitational potential from a disc mass map

    Args:
        mass: 2d array
        gravpot_kernel: 2d array (potential kernel)

    Returns:
        Potential from the convolution of the potential_kernel and mass
    """
    # Initialise array with zeroes
    return -Ggrav.value * convolve_fft(gravpot_kernel, mass)

def get_forces(xpc, ypc, gravpot):
    """Calculation of the forces
    """
    # Force from the gradient of the potential
    F_grad = np.gradient(gravpot)

    # Getting the polar coordinates
    R, theta = xy_to_polar(xpc, ypc)
    theta_rad = np.deg2rad(theta)
    stepx_pc = xpc[1] - xpc[0]
    stepy_pc = ypc[1] - ypc[0]

    # Force components in X and Y
    Fx = F_grad[1] / stepx_pc
    Fy = F_grad[0] / stepy_pc

    # Radial force vector in outward direction
    Frad =   Fx * np.cos(theta_rad) + Fy * np.sin(theta_rad)
    # Tangential force vector in clockwise direction
    Ftan = - Fx * np.sin(theta_rad) + Fy * np.cos(theta_rad)
    return Fgrad, Fx, Fy, Frad, Ftan

def get_vrot_from_force(rpc, Frad):
    """Calculate the rotation velocity from the radial forces

    Args:
        rpc:
        Frad:

    Returns:
        vrot
    """
    return np.sqrt(np.abs(rpc * Frad))

def get_torque(xpc, ypc, vel, Fx, Fy, weights, n_rbins):
    """Calculation of the gravity torques
    """
    # Torque is just Deprojected_Gas * (X * Fy - y * Fx)
    torque = (xpc * Fy - ypc * Fx) * weights

    # Average over azimuthal angle and normalization
    rpc = np.sqrt(xpc**2 + ypc**2)
    rsamp, stepr = get_rsamp(rpc, n_rbins)

    # And now binning with the various weights
    # Torque
    weights_mean = stats.bin_statistics(r, weights, statistics='mean', bins=rsamp)
    torque_mean = stats.bin_statistics(r, torque, statistics='mean', bins=rsamp)
    torque_mean_w = torque_mean / weights_mean

    # Angular momentum
    r_mean = stats.bin_statistics(r, r, statistics='mean', bins=rsamp)
    vel_mean = stats.bin_statistics(r, vel, statistics='mean', bins=rsamp)
    ang_mom_mean  = r_mean * vel_mean

    # Specific angular momentum in one rotation
    dl = torque_mean_w / ang_mom_mean

    # Mass inflow/outflow rate
    dm = dl * 2. * np.pi * r_mean * weights_mean

    # Mass inflow/outflow integrated over a certain radius R
    dm_sum = np.cumsum(dm) * stepr

    return torque_mean, torque_mean_w, r_mean, ang_mom_mean, dl, dm, dm_sum

