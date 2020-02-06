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
from os.path import join as joinpath

# local modules
from .galaxy import Galaxy
from .disc_data import Map, Profile
from .misc_io import add_err_prefix, AttrDict, read_vc_file
from .local_units import *

dic_moments = OrderedDict([
              # order   type   attr_name  comment
                (-1, [("Xin", "X", "X axis coordinate", pixel),
                      ("Yin", "Y", "Y axis coordinate", pixel)]),
                (0, [("Flux", "I", "flux [integrated]", Lsun),
                     ("Mass", "M", "mass [integrated]", Msun),
                     ("DFlux", "Id", "flux density [per unit area]", Lsunpc2),
                     ("DMass", "Md", "mass density [per unit area]", Msunpc2),]),
                (1, [("Vel", "V", "velocity", kms)]),
                (2, [("Disp", "S", "dispersion", kms),
                     ("Mu2", "mu2", "non-centred 2nd order moment", kms2)])
                ])

dic_invert_moments = {}
for order in dic_moments.keys():
    for tup in dic_moments[order]:
        dic_invert_moments[tup[0]] = (order,) + tup[1:]

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

def print_dict_maps():
    """List the potential Maps attributes from dic_moments
    Returns:
        names for the Maps
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

        # initialise Maps if any
        self._reset_maps()
        self._reset_profiles()
        self.add_map(**kwargs)

        # init slicing
        self._reset_slicing()

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

    def _reset_slicing(self):
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

    def _reset_maps(self):
        """Initalise the Maps by setting an empty
        'maps' dictionary
        """
        # set up the Maps
        self.maps = AttrDict()

    def _reset_profiles(self):
        """Initalise the Profiles by setting an empty
        'profiles' dictionary
        """
        # set up the Profiles
        self.profiles = AttrDict()

    @property
    def nprofiles(self):
        """Number of existing profiles
        """
        if hasattr(self, 'profiles'):
            return len(self.profiles)
        else:
            return -1

    @property
    def nmaps(self):
        """Number of existing maps
        """
        if hasattr(self, 'maps'):
            return len(self.maps)
        else:
            return -1

    def add_map(self, comment="", **kwargs):
        """Adding a Map which can include various keywords like
        Xin, Yin, Vel, etc.
        It is important to provide a flag so that the Map has a unique entry
        in the 'Map' dictionary.

        Args:
            comment: str
            **kwargs:
                name: str
                    Name of the Map
        """
        if self.nmaps <= 0:
            self._reset_maps()

        # First scan all the input kwargs to see which data are there
        # This will set up the attributes which are relevant
        ref_data_shape = self._get_ref_shape(**kwargs)
        # if _set_ref_data is not defined, it means we didn't detect
        # any proper numpy array
        if ref_data_shape is None:
            print("WARNING: no Map yet provided - model will be empty")
            print("       ==> Use model.add_map() to add a Map")
            return

        map_name = kwargs.pop("name", "Map{:02d}".format(self.nmaps + 1))
        if self._has_map(map_name):
            if not overwrite:
                print("ERROR[add_map]: trying to overwrite an existing Map. "
                      "Use overwrite=True if you wish to proceed with this. - Aborting -")
                return
            else:
                print("WARNING[add_map]: overwriting existing Map ({}).".format(map_name))

        # First initialise the Map with X and Y by reading
        # Xin and Yin as defined in the dictionary
        Xin = kwargs.pop(get_moment_type(-1)[0], None)
        Yin = kwargs.pop(get_moment_type(-1)[1], None)

        # The names of the given X and Y attribute names
        # are by default defined in the dictionary
        nameX = get_moment_attr(-1)[0]
        nameY = get_moment_attr(-1)[1]

        # Now create the corresponding Map
        newmap = Map(Xin, Yin, nameX=nameX, nameY=nameY,
                     ref_shape=ref_data_shape, comment=comment,
                     name=map_name, **kwargs)

        # If we have now a proper Map with a shape
        # We can add all the provided datamaps
        all_tuples = get_all_moment_tuples()
        for kwarg in kwargs:
            for tup in all_tuples:
                if kwarg.startswith(tup[0]):
                    # name of variable and attribute
                    dmap_flag = kwarg.replace(tup[0], "")
                    dmap_name = kwarg.replace(tup[0], tup[1])
                    dmap_type = tup[2]
                    order = dic_invert_moments[tup[0]][0]
                    unit = dic_invert_moments[tup[0]][3]

                    # Extracting the kwarg array data and edata
                    data = kwargs.get(kwarg, None)
                    edata = kwargs.get(add_err_prefix(kwarg), None)

                    # If data is not None and array
                    if isinstance(data, np.ndarray):
                        newmap.add_data(data, order=order, edata=edata,
                                        data_name=dmap_name, data_flag=dmap_flag,
                                        data_type=dmap_type, data_unit=unit)

        newmap.align_axes(self)
        self.maps[map_name] = newmap

    def _has_map(self, name):
        if self.nmaps <= 0:
            return False
        else:
            return name in self.maps.keys()

    def _get_datamap(self, map_name, datamap_name=None, order=None):
        """Get the datamap from a given Map
        """
        # Get the Map
        ds = self._get_map(map_name)
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
            print("No such datamap {} in this Map".format(
               datamap_name))
            return ds, None
        return ds, datamap

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

    def _get_map(self, map_name=None, type=""):
        """
        Args:
            **kwargs:
        Returns:
            Either Map if provided, otherwise
            just the first Map name.
        """
        # if name is None, get the first Map
        if map_name is None :
            for map_name in self.maps.keys():
                if type in self.maps[map_name].type:
                    break
        return self.maps[map_name]

    def _get_profile(self, profile_name=None, type=""):
        """
        Args:
            **kwargs:
        Returns:
            Either profile_name if provided, otherwise
            just the first profile name.
        """
        # if name is None, get the first Profile
        if profile_name is None :
            for profile_name in self.profiles.keys():
                if type in self.profiles[profile_name].type:
                    break
        return self.profiles[profile_name]

    def add_vprofile(self, filename=None, filetype="ROTCUR", folder="",
                     prof_name=None, **kwargs):
        """Reading the input V file

        Input
        -----
        filename: str
            Name of the Vcfile
        filetype: str
            'ROTCUR' or 'ASCII'
        folder: str
        name: str
        """
        if filename is None:
            print("ERROR: no filename provided - Aborting")

        if self.verbose :
            print("Reading the V file")

        # Reading of observed rot velocities
        filename = joinpath(folder + filename)
        status, R, Vc, eVc = read_vc_file(filename=filename, filetype=filetype)

        if status == 0:
            if self.nprofiles < 0:
                self._reset_profiles()
            if prof_name is None:
                prof_name = "Vprof_{:02d}".format(self.nprofiles + 1)
            self.profiles[prof_name] = Profile(data=Vc, edata=eVc, Rin=R,
                                          data_name=prof_name, order=1,
                                          type='vel', data_unit=kms,
                                          data_type=filetype, **kwargs)
            if self.verbose:
                print("Vc file successfully read")
        else:
            print("ERROR status {}".format(status))

    def deproject_nodes(self, map_name=None):
        """Deproject disc mass or flux
        """
        self._get_map(map_name).deproject(self)

    def deproject_vprofile(self, profile_name):
        """Deproject Velocity values by dividing by the sin(inclination)
        """
        if profile_name not in self.profiles.keys():
            print("ERROR: no such profile ({}) in this model - Aborting".format(
                  profile_name))
            return
        if self.profiles[profile_name].order != 1:
            print("ERROR[deproject_vprofile]: profile not of order=1 - Aborting")
            return

        self.profiles[profile_name].deproject_velocities(self.inclin)

    def get_radial_profile(self, map_name, datamap_name=None, order=0):
        """Get a radial profile from a given map. If datamap_name
        is None, it will use the first map of order 0.

        Args:
            map_name:
            datamap_name:
            order:

        Returns:

        """
        pass



