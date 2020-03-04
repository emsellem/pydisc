# -*- coding: utf-8 -*-
"""
This provides the Density Wave functionalities, as a class inheriting from GalacticDisc
"""
import numpy as np

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

from .disc import GalacticDisc
from .disc_data import Slicing
from .misc_io import add_suffix
from .plotting import show_tw
from .check import _none_tozero_array
from .fit_functions import fit_slope

# Units
from . import local_units as lu

class DensityWave(GalacticDisc):
    """
    Main DensityWave class, describing a galactic disc with some Wave propagating
    and useful methods (e.g., Tremaine Weinberg).

    Attributes
    ----------

    """
    def __init__(self, force_dtype=True, **kwargs):
        """

        Args:
            force_dtype (bool): forcing dtype or not [True]
        """
        self.verbose = kwargs.pop("verbose", False)

        # Using GalacticDisc class attributes
        super().__init__(force_dtype=force_dtype, **kwargs)

    def get_bar_VRtheta(self, mname=None):
        """Compute the in-plane deprojected velocities for a barred
        system, using a mirror technique developed by Witold Maciejewski.

        Parameters
        ----------
        mname (str): Name of the map to use
       """
        ds = self._get_map(mname)
        self.deproject_velocities(mname)
        ds.align_xy_deproj_bar(self)

        # Mirroring the Velocities
        ds.V_mirror = gdata(np.vstack((ds.X.ravel(), ds.Y.ravel())).T,
                                 ds.Vdep.ravel(),
                                 (ds.X_mirror, ds.Y_mirror),
                                 fill_value=ds._fill_value, method=ds._method)
        ds.gamma_rad = np.arctan2(ds.Y_bardep, ds.X_bardep)
        ds.Vr = (ds.Vdep * cos(self._PAbar_londep_rad - ds.gamma_rad)
                - ds.V_mirror * cos(self._PAbar_londep_rad + ds.gamma_rad)) \
                  / sin(2.* self._PAbar_londep_rad)
        ds.Vt = (ds.Vdep * sin(self._PAbar_londep_rad - ds.gamma_rad)
                + ds.V_mirror * sin(self._PAbar_londep_rad + ds.gamma_rad)) \
                  / sin(2.* self._PAbar_londep_rad)
        ds.Vx = ds.Vr * cos(ds.gamma_rad) - ds.Vt * sin(ds.gamma_rad)
        ds.Vy = ds.Vr * sin(ds.gamma_rad) + ds.Vt * cos(ds.gamma_rad)

    def tremaine_weinberg(self, flux_name="flux", vel_name="vel",
                          slit_width=1.0, mname=None,
                          **kwargs):
        """ Apply the standard Tremaine Weinberg to the disc Map.

        Using X_lon, Y_lon, Flux and Velocity

        Input
        -----
        slit_width (float): Slit width in arcsecond [1.0]
        flux_name (str): ['flux']
        vel_name (str): ['vel']
            names of the datamaps where to find the tracer (flux or velocities)
        """
        # Getting the map from the name. If name is None, use the 1st one
        ds = self._get_map(mname)
        # Align the axes
        ds.align_xy_lineofnodes(self)
        # Get the unit for X, Y
        uXY = ds.XYunit

        # Getting the maps (tracer flux, and velocities)
        Fmap = ds.dmaps[flux_name]
        Vmap = ds.dmaps[vel_name]
        Flux, eFlux, uFlux = Fmap.data, Fmap.edata, Fmap.dunit
        Vel, eVel, uVel = Vmap.data, Vmap.edata, Vmap.dunit

        # Check the uncertainties if not given
        eVel = _none_tozero_array(eVel, Vel)
        eFlux = _none_tozero_array(eFlux, Flux)
        if eVel is None or eFlux is None:
            print("ERROR: could not format error array - Aborting")
            return

        # Converting X in kpc
        fac_kpc, newXY_unit = self.convert_xyunit(uXY)
        X_lon_kpc = ds.X_lon * fac_kpc
        uXY = newXY_unit

        # Get Flux * Velocities
        fV = Flux * -Vel
        ufV = uFlux * uVel
        # Get the Flux * X
        fx = Flux * X_lon_kpc
        ufx = uFlux * uXY
        # Get the errors
        fV_err = fV * np.sqrt((eFlux / Flux)**2 + (eVel / Vel)**2)

        ds_slits = Slicing(yin=ds.Y_lon, slit_width=slit_width)
        ds_slits.flux_dmapname = flux_name
        ds_slits.vel_dmapname = vel_name
        # Digitize the Y coordinates along the slits and minus 1 to be at the boundary
        dig = np.digitize(ds.Y_lon, ds_slits.yedges).ravel() - 1
        # Select out points which are out of the edges
        selin = (dig >= 0) & (dig < len(ds_slits.yedges)-1)

        # Then count them with the weights
        flux_slit = np.bincount(dig[selin],
                                weights=np.nan_to_num(Flux).ravel()[selin])
        fluxVel_slit = np.bincount(dig[selin],
                                   weights=np.nan_to_num(fV).ravel()[selin])
        fluxX_slit = np.bincount(dig[selin],
                                 weights=np.nan_to_num(fx).ravel()[selin])

        # Do the calculation and get the right unit
        conv_factor, Om_unit = lu.get_conversion_factor(ufV / ufx, lu.kmskpc)
        ds_slits.Omsini_tw = fluxVel_slit * conv_factor / fluxX_slit
        ds_slits.unit_Omsini_tw = Om_unit

        ds_slits.dfV_tw = fluxVel_slit / flux_slit
        ds_slits.dfx_tw = fluxX_slit / flux_slit

        # Calculate errors.
        err_flux_slit = np.sqrt(np.bincount(dig[selin],
                                            weights=np.nan_to_num(eFlux**2).ravel()[selin]))
        err_fluxVel_slit = np.sqrt(np.bincount(dig[selin],
                                               weights=np.nan_to_num(fV_err**2).ravel()[selin]))
        err_percentage_vel = err_fluxVel_slit / fluxVel_slit
        err_percentage_flux = err_flux_slit / flux_slit

        ds_slits.dfV_tw_err = np.abs(ds_slits.dfV_tw) * np.sqrt(err_percentage_vel**2
                                                                + err_percentage_flux**2)
        ds_slits.dfx_tw_err = np.abs(ds_slits.dfx_tw) * err_percentage_flux
        ds_slits.Omsini_tw_err = ds_slits.Omsini_tw * np.sqrt((ds_slits.dfV_tw_err / ds_slits.dfV_tw)**2
                                                              + (ds_slits.dfx_tw_err / ds_slits.dfx_tw)**2)

        self.add_slicing(ds_slits, ds.mname)

    def fit_slope_tw(self, slicing_name=None, select_num=[]):
        """Fitting the slope of the Tremaine Weinberg method
        """
        # get the slicing
        ds_slits = self._get_slicing(slicing_name)
        if ds_slits is None:
            return None
        if select_num == []:
            sel = np.ones_like(ds_slits.dfx_tw, dtype=np.bool)
        else:
            sel = np.array(select_num).astype(np.int)
        if ds_slits.dfx_tw_err is None or np.all(ds_slits.dfx_tw_err == 0):
            e_dfx = None
        else:
            e_dfx = ds_slits.dfx_tw_err[sel]
        if ds_slits.dfV_tw_err is None or np.all(ds_slits.dfV_tw_err == 0):
            e_dfV = None
        else:
            e_dfV = ds_slits.dfV_tw_err[sel]
        return fit_slope(ds_slits.dfx_tw[sel], ds_slits.dfV_tw[sel],
                         e_dfx, e_dfV)

    def plot_tw(self, slicing_name=None, **kwargs):
        """Plot the results from the Tremaine Weinberg method.

        Args:
            slicing_name: str [None]
            **kwargs: see plot_tw in plotting module.

        Returns:

        """
        show_tw(self, slicing_name, **kwargs)