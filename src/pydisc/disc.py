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
from .disc_data import DataSet, DataMap
from .misc_io import add_err_prefix
from .plotting import show_tw

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
    """List the potential DataSets attributes from dic_moments
    Returns:
        names for the DataSets
    """
    for order in dic_moments.keys():
        for defset in dic_moments[order]:
            print("{0:10}: {1:30} - Attribute={2:>5} [order {3:2}]".format(
                    defset[0], defset[2], defset[1], order))

class GalacticDisc(Galaxy):
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

        # initialise DataSets
        self.init_datasets(**kwargs)

        # init slicing
        self.init_slicing()

    @property
    def slicing(self):
        if bool(self.slicings):
            return self.slicings[list(self.slicings.keys())[0]]
        else:
            return {}

    def _get_slicing(self, slicing_name):
        if slicing_name is None:
            return self.slicing
        else:
            return self.slicings[slicing_name]

    def init_slicing(self):
        """Initialise the slice dictionary if needed
        """
        if not hasattr(self, "slicings"):
            self.slicings = {}

    def add_slicing(self, value, slicing_name=""):
        """
        Args:
            slits_name: str
                Name of the slicing
            slits: Slicing
                Input to add.
        """
        self.slicings[slicing_name] = value

    def init_datasets(self, **kwargs):
        """Initalise the DataSet by setting an empty
        'DataSets' dictionary, and filling it in with the
        provided input.
        Will then use 'add_DataSet' to add a given set of maps.

        Args:
            **kwargs:
        """
        # set up the DataSets
        self.datasets = {}
        self.add_dataset(**kwargs)

    def add_dataset(self, flag=None, comment="", **kwargs):
        """Adding a DataSet which can include various keywords like
        Xin, Yin, Vel, etc.
        It is important to provide a flag so that the DataSet has a unique entry
        in the 'DataSets' dictionary.

        Args:
            **kwargs:
        """
        # First scan all the input kwargs to see which data are there
        # This will set up the attributes which are relevant
        ref_data_shape = self._get_ref_shape(**kwargs)
        # if _set_ref_data is not defined, it means we didn't detect
        # any proper numpy array
        if ref_data_shape is None:
            print("WARNING: no proper DataSet was provided - model will be empty")
            print("       ==> Use model.init_data() to initialise the DataSets")
            return

        dataset_name = kwargs.pop("dataset_name", "dataset{:02d}".format(len(self.datasets) + 1))

        # First initialise the DataSet with X and Y
        Xin = kwargs.get(read_mom_attr(-1)[0], None)
        Yin = kwargs.get(read_mom_attr(-1)[1], None)
        nameX = read_mom_sattr(-1)[0]
        nameY = read_mom_sattr(-1)[1]
        mydataset = DataSet(Xin, Yin, nameX=nameX, nameY=nameY,
                            ref_shape=ref_data_shape, comment=comment,
                            dataset_name=dataset_name, flag=flag)

        # If we have now a proper DataSet with a shape
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
                    mydataset.attach_datamap(data, order=order, edata=edata,
                                          data_name=data_name,
                                          data_attr_name=kwarg_name)

        self.datasets[dataset_name] = mydataset

    def _get_datamap(self, dataset_name, datamap_name=None, order=None):
        """Get the datamap from a given dataset
        """
        # Get the dataset
        ds = self._get_dataset(dataset_name)
        if datamap_name is None:
            if order is None:
                # then just get the first map
                datamap_name = list(ds.datamaps.keys())[0]
            else:
                # Then get the first map of right order
                for key in ds.datamaps.keys():
                    if ds.datamaps[key].order == order:
                        datamap_name = key
                        break

        if hasattr(ds, datamap_name):
            datamap = ds.datamap_name
        else:
            print("No such datamap {} in this dataset".format(
               datamap_name))
            return ds, datamap_name, None
        return ds, datamap_name, datamap

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
        # if name is None, get the first DataSet
        if dataset_name is None :
            dataset_name = list(self.datasets.keys())[0]
        return self.datasets[dataset_name]

    def add_Vc_profile(self, vc_filename=None, vc_filetype="ROTCUR"):
        """Reading the input Vc file

        Input
        -----
        vc_filename: str
            Name of the Vcfile

        vc_filetype: str
            'ROTCUR' or 'ASCII'

        Returns
        -------
        status: int
            0 means it was read. -1: the file does not exist
            -2: file type not recognised.
        """
        if vc_filename is None:
            if self.vc_filename is None:
                print("ERROR: no Vc filename provided")
                return -1

            vc_filename = self.vc_filename
        if self.verbose :
            print("Reading the Vc file")

        ##--- Reading of observed rot velocities
        vc_filename = joinpath(self.folder + vc_filename)
        status, self.Vcobs_r, self.Vcobs, self.eVcobs, \
                self.Vcobs_rint, self.Vcobs_int = read_vcirc_file(vc_filename,
                        vc_filetype=vc_filetype)

        if status == 0 * self.verbose:
            print("Vc file successfully read")
        return status

    def deproject_disc_map(self, dataset_name=None, datamap_name=None):
        """Deproject disc mass or flux
        """
        ds, key_dm, dm = self._get_datamap(dataset_name, datamap_name, order=0)
        if dm is None:
            print("ERROR: maps not available")
            return

        ds[key_dm].data_dep = transform.deproject_frame(dm.data, PA=self.PA_nodes, inclination=self.inclin)

    def deproject_velocity_profile(self, dataset_name=None, datamap_name=None):
        """Deproject Velocity values by dividing by the sin(inclination)
        """

        ds, key_dm, dm = self._get_datamap(dataset_name, datamap_name, order=1)
        if dm is None:
            print("ERROR: Velocity datamap not found in this dataset")
            return

        dep_key = "_dep".format(key_dm)
        dep_name = "_dep".format(dm.name)
        Vdep, eVdep = transform.deproject_velocity_profile(dm.data, dm.edata, self.inclin)
        ds.attach_datamap(Vdep, order=1, edata=eVdep, data_name=dep_name, data_attr_name=dep_key)
