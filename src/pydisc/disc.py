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
from .disc_data import DataSet, DataProfile
from .misc_io import add_err_prefix, AttrDict

dic_moments = OrderedDict([
              # order   type   attr_name  comment
                (-1, [("Xin", "X", "X axis coordinate"),
                      ("Yin", "Y", "Y axis coordinate")]),
                (0, [("Flux", "I", "flux [integrated]"),
                     ("Mass", "M", "mass [integrated]"),
                     ("DFlux", "Id", "flux density [per unit area]"),
                     ("DMass", "Md", "mass density [per unit area]"),]),
                (1, [("Vel", "V", "velocity")]),
                (2, [("Disp", "S", "dispersion"),
                     ("Mu2", "mu2", "non-centred 2nd order moment")])
                ])

dic_invert_moments = {}
for order in dic_moments.keys():
    for tup in dic_moments[order]:
        dic_invert_moments[tup[0]] = order

def get_all_moment_types():
    """Get all types for all moments
    """
    alltypes = []
    for order in dic_moments.keys():
        alltypes.extend(get_moment_type(order))

    return alltypes

def get_all_moment_tuples():
    """Get all tuples from dic_moments
    """
    alltuples = []
    for order in dic_moments.keys():
        alltuples.extend(dic_moments[order])

    return alltuples

def get_moment_type(order):
    """Returns all potential variable types for this order

    Args
    ----
    order: int
    """
    return [defset[0] for defset in dic_moments[order]]

def get_moment_attr(order):
    """Returns all potential attribute names for this order
    
    Args
    ----
    order: int
    """
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
        """Initialise the GalacticDisc class, first by initialising the
        Galaxy class.

        Args:
            verbose: bool [False]
            **kwargs:
                distance
                pc_per_arcsec
                inclin
                PA_nodes
                PA_bar
        """
        self.verbose = kwargs.pop("verbose", False)

        # Using galaxy class attributes
        super().__init__(**kwargs)

        # initialise DataSets if any
        self.add_dataset(**kwargs)

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

    def _reset_datasets(self):
        """Initalise the DataSet by setting an empty
        'DataSets' dictionary
        """
        # set up the DataSets
        self.datasets = AttrDict()

    def _reset_vprofiles(self):
        """Initalise the vprofiles by setting an empty
        dictionary
        """
        # set up the DataSets
        self.Vprofiles = AttrDict()

    @property
    def nVprofiles(self):
        if hasattr(self, 'Vprofiles'):
            return len(self.Vprofiles)
        else:
            return -1

    @property
    def ndatasets(self):
        if hasattr(self, 'datasets'):
            return len(self.datasets)
        else:
            return -1

    def add_dataset(self, comment="", **kwargs):
        """Adding a DataSet which can include various keywords like
        Xin, Yin, Vel, etc.
        It is important to provide a flag so that the DataSet has a unique entry
        in the 'DataSets' dictionary.

        Args:
            comment: str
            **kwargs:
                name: str
                    Name of the dataset
        """
        if self.ndatasets <= 0:
            self._reset_datasets()

        # First scan all the input kwargs to see which data are there
        # This will set up the attributes which are relevant
        ref_data_shape = self._get_ref_shape(**kwargs)
        # if _set_ref_data is not defined, it means we didn't detect
        # any proper numpy array
        if ref_data_shape is None:
            print("WARNING: no DataSet yet provided - model will be empty")
            print("       ==> Use model.add_dataset() to add a dataset")
            return

        dataset_name = kwargs.pop("name", "dataset{:02d}".format(self.ndatasets + 1))

        # First initialise the DataSet with X and Y by reading
        # Xin and Yin as defined in the dictionary
        Xin = kwargs.get(get_moment_type(-1)[0], None)
        Yin = kwargs.get(get_moment_type(-1)[1], None)

        # The names of the given X and Y attribute names
        # are by default defined in the dictionary
        nameX = get_moment_attr(-1)[0]
        nameY = get_moment_attr(-1)[1]

        # Now create the corresponding DataSet
        mydataset = DataSet(Xin, Yin, nameX=nameX, nameY=nameY,
                            ref_shape=ref_data_shape, comment=comment,
                            name=dataset_name, **kwargs)

        # If we have now a proper DataSet with a shape
        # We can add all the provided data
        all_tuples = get_all_moment_tuples()
        for kwarg in kwargs:
            for tup in all_tuples:
                if kwarg.startswith(tup[0]):
                    # name of variable and attribute
                    map_flag = kwarg.replace(tup[0], "")
                    map_name = kwarg.replace(tup[0], tup[1])
                    map_type = tup[2]
                    order = dic_invert_moments[tup[0]]

                    # Extracting the kwarg array
                    data = kwargs.get(kwarg, None)
                    # See if uncertainties are there
                    edata = kwargs.get(add_err_prefix(kwarg), None)
                    if edata is None:
                        edata = np.zeros_like(data)

                    # If data is not None and array
                    if isinstance(data, np.ndarray):
                        mydataset.attach_datamap(data, order=order, edata=edata,
                                                 map_name=map_name,
                                                 map_flag=map_flag,
                                                 map_type=map_type)

        mydataset.align_axes(self)
        self.datasets[dataset_name] = mydataset

    def _has_dataset(self, name):
        if self.ndatasets <= 0:
            return False
        else:
            return name in self.datasets.keys()

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
            datamap = getattr(ds, datamap_name)
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
                        break
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

    def add_vprofile(self, vfilename=None, vfiletype="ROTCUR", folder="", flag=None):
        """Reading the input V file

        Input
        -----
        vfilename: str
            Name of the Vcfile
        vfiletype: str
            'ROTCUR' or 'ASCII'
        folder: str
        flag: str
        """
        if vfilename is None:
            print("ERROR: no Vfilename provided - Aborting")

        if self.verbose :
            print("Reading the V file")

        # Reading of observed rot velocities
        vfilename = joinpath(folder + vfilename)
        status, R, Vc, eVc = read_vc_file(vfilename,
                        vc_filetype=vfiletype)

        if status == 0:
            if self.nVprofiles < 0:
                self._reset_vprofiles()
            if flag is None:
                flag = "Vprof_{:02d}".format(self.nVprofiles + 1)
            self.Vprofiles[flag] = DataProfile(Vc, eVc, radii=R, name=flag, order=1, type=vc_filetype)
            if self.verbose:
                print("Vc file successfully read")
        else:
            print("ERROR status {}".format(status))

    def deproject_nodes(self, dataset_name=None):
        """Deproject disc mass or flux
        """
        ds = self._get_dataset(dataset_name)
        ds.deproject(self)

    def deproject_vprofile(self, profile_name):
        """Deproject Velocity values by dividing by the sin(inclination)
        """
        if profile_name not in self.Vprofiles.keys():
            print("ERROR: no such profile ({}) in this model - Aborting".format(
                  profile_name))

        self.Vprofiles[profile_name].deproject_velocities(self.inclin)

