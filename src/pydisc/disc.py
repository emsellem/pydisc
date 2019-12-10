# -*- coding: utf-8 -*-
"""
This is the main class of the package - disc - gathering
data, attributes and functions defining a disc model.

This module include computation for torques, Tremaine-Weinberg
method, in-plane velocities (Maciejewski et al. method).
"""

# general modules
from collections import OrderedDict

# External modules
import numpy as np
from numpy import cos, sin
from scipy.interpolate import griddata as gdata

# local modules
from .galaxy import Galaxy
from . import transform
from .disc_data import dataset
from .misc_io import add_err_prefix

dic_moments = OrderedDict([
                (-1, [("Xin", "X", "X axis coordinate"), ("Yin", "Y", "Y axis coordinate")]),
                (0, [("Flux", "I", "flux"), ("Mass", "M", "mass"),
                     ("FluxD", "Id", "flux density"), ("MassD", "Md", "mass density")]),
                (1, [("Vel", "V", "velocity")]),
                (2, [("Disp", "S", "dispersion"), ("Mu2", "mu2", "non-centred 2nd order moment")])
                ])

def read_mom_attr(order):
    return [defset[0] for defset in dic_moments[order]]

def read_mom_sattr(order):
    return [defset[1] for defset in dic_moments[order]]

def print_datasets():
    """List the potential datasets attributes from dic_moments
    Returns:
        names for the datasets
    """
    for order in dic_moments.keys():
        for defset in dic_moments[order]:
            print("{0:10}: {1:30} - Attribute={2:>5} [order {3:2}]".format(
                    defset[0], defset[2], defset[1], order))

class DiscModel(Galaxy):
    """
    Main discmodel class, describing a galactic disc and providing
    computational functions.

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

        # Using galaxy class attributes
        super().__init__(**kwargs)

        # initialise datasets
        self.init_datasets(**kwargs)

    def init_datasets(self, **kwargs):
        """Initalise the dataset by setting an empty
        'datasets' dictionary, and filling it in with the
        provided input.
        Will then use 'add_dataset' to add a given set of maps.

        Args:
            **kwargs:
        """
        # set up the datasets
        self.datasets = {}
        self.add_dataset(**kwargs)

    def add_dataset(self, flag=None, comment="", **kwargs):
        """Adding a dataset which can include various keywords like
        Xin, Yin, Vel, etc.
        It is important to provide a flag so that the dataset has a unique entry
        in the 'datasets' dictionary.

        Args:
            **kwargs:
        """
        # First scan all the input kwargs to see which data are there
        # This will set up the attributes which are relevant
        ref_data_shape = self._get_ref_shape(**kwargs)
        # if _set_ref_data is not defined, it means we didn't detect
        # any proper numpy array
        if ref_data_shape is None:
            print("WARNING: no proper dataset was provided - model will be empty")
            print("       ==> Use model.init_data() to initialise the datasets")
            return

        dataset_name = kwargs.pop("dataset_name", "dataset{:02d}".format(len(self.datasets) + 1))

        # First initialise the dataset with X and Y
        Xin = kwargs.get(read_mom_attr(-1)[0], None)
        Yin = kwargs.get(read_mom_attr(-1)[1], None)
        nameX = read_mom_sattr(-1)[0]
        nameY = read_mom_sattr(-1)[1]
        mydataset = dataset(Xin, Yin, nameX=nameX, nameY=nameY,
                            ref_shape=ref_data_shape, comment=comment,
                            dataset_name=dataset_name, flag=flag)

        # If we have now a proper dataset with a shape
        # We can add all the provided data
        for order in dic_moments.keys():
            for desc_data in dic_moments[order]:
                # Name of input parameter
                kwarg_name = desc_data[0]
                # name of the attribute
                data_name = desc_data[1]

                # Extracting the kwarg array
                data = kwargs.pop(kwarg_name, None)
                # If order is higher than 0, then see if error is there
                edata = kwargs.pop(add_err_prefix(kwarg_name), None)

                # If data is not None and array
                if isinstance(data, np.ndarray):
                    mydataset.attach_data(data, order=order, edata=edata,
                                          data_name=data_name,
                                          data_attr_name=kwarg_name)

        self.datasets[dataset_name] = mydataset

    def _get_ref_shape(self, **kwargs):
        # Getting all the input by scanning attribute names
        # in the input dic_moments dictionary
        ref_data_shape = None
        for order in dic_moments.keys():
            for desc_data in dic_moments[order]:
                kwarg_name = desc_data[0]
                data = kwargs.get(kwarg_name, None)
                if data is not None:
                    if isinstance(data, (np.ndarray)):
                        ref_data_shape = data.shape
                    else:
                        print("WARNING: data {} not a numpy array".format(
                                kwarg_name))

        return ref_data_shape

    def _get_dataset(self, dataset_name=None):
        """
        Args:
            **kwargs:
        Returns:
            Either dataset_name if provide, otherwise
            just the first dataset name.
        """
        # if name is None, get the first dataset
        if dataset_name is None :
            dataset_name = list(self.datasets.keys())[0]
        return self.datasets[dataset_name]

    def deproject_velocities(self, dataset_name=None):
        """Deproject Velocity values by dividing by the sin(inclination)
        """

        dataset = self._get_dataset(dataset_name)
        if hasattr(dataset, read_mom_attr(1)[0]):
            V = getattr(dataset, read_mom_attr(1)[0])
            dataset.Vdep = transform.deproject_velocities(V, self.inclin)
        else:
            print("ERROR: no velocity data found in this dataset")

    def get_Vrt(self, dataset_name=None):
        """Compute the in-plane deprojected velocities

        Input
        -----
        dataset_name: str
            Name of the dataset to use
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

    def tremaine_weinberg(self, slit_width=1.0, dataset_name=None, op=1):
        """ Get standard Tremaine Weinberg method applied on the bar

        Using X_lon, Y_lon, Flux and Velocity
        """
        ds = self._get_dataset(dataset_name)
        ds.align_xy_lineofnodes(self)

        # Get Flux * Velocities
        fV = ds.Flux * -ds.Vel
        # Get the Flux * X
        fx = ds.Flux * ds.X_lon

        # Get the errors
        fV_err = fV * np.sqrt((ds.eFlux / ds.Flux)**2 + (ds.eVel / ds.Vel)**2)

        maxY = np.maximum(np.abs(np.max(ds.Y_lon)), np.abs(np.min(ds.Y_lon)))
        self.nslits = np.int(maxY / slit_width - 1 / 2.)
        self.y_slits = np.linspace(-self.nslits * slit_width, self.nslits * slit_width,
                                    2*self.nslits+1)
        sw2 = slit_width / 2.

        # Digitize the Y coordinates along the slits
        dig = np.digitize(ds.Y_lon, self.y_slits).ravel()

        # Then count them with the weights
        flux_slit = np.bincount(dig, weights=np.nan_to_num(ds.Flux).ravel())
        fluxVel_slit = np.bincount(dig, weights=np.nan_to_num(fV).ravel())
        fluxX_slit = np.bincount(dig, weights=np.nan_to_num(fx).ravel())
        self.Omsini_tw = fluxVel_slit / fluxX_slit
        self.dfV_tw = fluxVel_slit / flux_slit
        self.dfx_tw = fluxX_slit / flux_slit

        # Calculate errors.
        err_flux_slit = np.sqrt(np.bincount(dig, weights=np.nan_to_num(ds.eFlux**2).ravel()))
        err_fluxVel_slit = np.sqrt(np.bincount(dig, weights=np.nan_to_num(fV_err**2).ravel()))
        err_percentage_vel = err_fluxVel_slit / flux_slit
        err_percentage_flux = err_flux_slit / flux_slit

        self.dfV_tw_err = np.abs(self.dfV_tw) * np.sqrt(err_percentage_vel**2 + err_percentage_flux**2)
        self.dfx_tw_err = np.abs(self.dfx_tw) * err_percentage_flux
        self.Omsini_tw_err = self.Omsini_tw * np.sqrt((self.dfV_tw_err / self.dfV_tw)**2
                                                    + (self.dfx_tw_err / self.dfx_tw)**2)

#        # For a number of slits going from 0 to max
#        for i, y in enumerate(self.y_slits):
#            edges = [y - sw2, y + sw2]
#            selY = (ds.Y_lon > edges[0]) & (ds.Y_lon < edges[1])
#            self.df_tw[i] = np.nansum(ds.Flux[selY])
#            if self.df_tw[i] != 0:
#                self.dfV_tw[i] = np.nansum(fV[selY]) / self.df_tw[i]
#                self.dfx_tw[i] = np.nansum(fx[selY]) / self.df_tw[i]
#            else:
#                self.dfV_tw[i] = 0.
#                self.dfx_tw[i] = 0.
#
#            if self.dfx_tw[i] != 0:
#                self.Omsini_tw[i] = self.dfV_tw[i] / self.dfx_tw[i]
#            else:
#                self.Omsini_tw[i] = 0.
