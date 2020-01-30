# -*- coding: utf-8 -*-
"""
This provides the Density Wave functionalities, as a class inheriting from GalacticDisc
"""
import numpy as np
import os

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

from .disc import GalacticDisc
from .disc_data import Slicing
from .misc_io import add_suffix

class DensityWave(GalacticDisc):
    """
    Main DensityWave class, describing a galactic disc with some Wave propagating
    and useful methods (e.g., Tremaine Weinberg).

    Attributes
    ----------

    """
    def __init__(self, **kwargs):
        """

        Args:
            verbose: bool.
            data_name:
            flag:
        """
        self.verbose = kwargs.pop("verbose", False)

        # Using GalacticDisc class attributes
        super().__init__(**kwargs)

    def get_bar_VRtheta(self, dataset_name=None):
        """Compute the in-plane deprojected velocities for a barred
        system, using a mirror technique developed by Witold Maciejewski.

        Input
        -----
        dataset_name: str
            Name of the DataSet to use
       """
        ds = self._get_dataset(dataset_name)
        self.deproject_velocities(dataset_name)
        ds.align_xy_deproj_bar(self)

        ## Mirroring the Velocities
        ds.V_mirror = gdata(np.vstack((ds.Xin.ravel(), ds.Yin.ravel())).T,
                                 ds.Vdep.ravel(),
                                 (ds.X_mirror, ds.Y_mirror),
                                 fill_value=ds._fill_value, method=ds._method)
        ds.gamma_rad = np.arctan2(ds.Y_bardep, ds.X_bardep)
        ds.Vr = (ds.Vdep * cos(self._PA_barlon_dep_rad - ds.gamma_rad)
                - ds.V_mirror * cos(self._PA_barlon_dep_rad + ds.gamma_rad)) \
                  / sin(2.* self._PA_barlon_dep_rad)
        ds.Vt = (ds.Vdep * sin(self._PA_barlon_dep_rad - ds.gamma_rad)
                + ds.V_mirror * sin(self._PA_barlon_dep_rad + ds.gamma_rad)) \
                  / sin(2.* self._PA_barlon_dep_rad)
        ds.Vx = ds.Vr * cos(ds.gamma_rad) - ds.Vt * sin(ds.gamma_rad)
        ds.Vy = ds.Vr * sin(ds.gamma_rad) + ds.Vt * cos(ds.gamma_rad)

    def tremaine_weinberg(self, slit_width=1.0, dataset_name=None,
                          flag=None, **kwargs):
        """ Apply the standard Tremaine Weinberg to the disc dataset.

        Using X_lon, Y_lon, Flux and Velocity

        Input
        -----
        slit_width : float [1.0]
            Slit width in arcseconds.
        """
        ds = self._get_dataset(dataset_name)
        ds.align_xy_lineofnodes(self)

        Iname = kwargs.pop("Iname", add_suffix("I", flag))
        Vname = kwargs.pop("Vname", add_suffix("V", flag))

        Flux = getattr(ds.datamaps, Iname).data
        eFlux = getattr(ds.datamaps, Iname).edata
        Vel = getattr(ds.datamaps, Vname).data
        eVel = getattr(ds.datamaps, Vname).edata
        # Get Flux * Velocities
        fV = Flux * -Vel
        # Get the Flux * X
        fx = Flux * ds.X_lon
        # Get the errors
        fV_err = fV * np.sqrt((eFlux / Flux)**2 + (eVel / Vel)**2)

        ds_slits = Slicing(yin=ds.Y_lon, slit_width=slit_width)
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
        ds_slits.Omsini_tw = fluxVel_slit / fluxX_slit
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

        self.add_slicing(ds_slits, ds.name)

    def fit_slope_tw(self):
        pass

    def plot_tw(self, slicing_name=None, **kwargs):
        """Plot the results from the Tremaine Weinberg method.

        Args:
            slicing_name: str [None]
            **kwargs: see plot_tw in plotting module.

        Returns:

        """
        show_tw(self, slicing_name, **kwargs)