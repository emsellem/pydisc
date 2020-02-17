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
from .disc_data import Map, Profile, list_Data_attr
from .misc_io import add_err_prefix, add_suffix, AttrDict, read_vc_file
from .misc_io import extract_suffixes_from_kwargs as ex_suff
from .local_units import *

dic_moments = OrderedDict([
              # order   mtype   attr_name  comment unit
                (-10, [("Dummy", "dummy", "Dummy category", pixel)]),
                (0, [("Flux", "I", "flux [integrated]", Lsun),
                     ("Mass", "M", "mass [integrated]", Msun),
                     ("FluxD", "Id", "flux density [per unit area]", Lsunpc2),
                     ("MassD", "Md", "mass density [per unit area]", Msunpc2),
                     ("WeightD", "Wd", "To be used as weights [per unit area]", Lsunpc2),
                     ("Weight", "W", "To be used as weights [integrated]", Lsun)]),
                (1, [("Vel", "V", "velocity", kms)]),
                (2, [("Disp", "S", "dispersion", kms),
                     ("Mu2", "Mu2", "non-centred 2nd order moment", kms2)])
                ])

dict_invert_moments = {}
for order in dic_moments.keys():
    for tup in dic_moments[order]:
        dict_invert_moments[tup[0].lower()] = (order,) + tup[1:]

# List of attributes to use for Maps and DataMaps
list_Map_attr = ["Xcen", "Ycen", "X", "Y", "name",
                 "mtype", "XYunit", "pixel_scale",
                 "NE_direct", "alpha_North", "fill_value", "method",
                 "overwrite"]
list_Profile_attr = ["R", "name", "ptype", "nameR", "Runit"]

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

        self.force_dtypes = kwargs.pop("force_dtypes", True)
        read_maps = kwargs.pop("read_maps", True)
        if read_maps:
            self.add_maps(**kwargs)

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

    def _extract_dmap_info_from_dtype(self, dtype, dname=None):
        """Extract info using the current kwarg
        and the list of pre-defined tuples

        Args:
            dtype:

        Returns:
            datamap name, flag, dtype, order, dunit
        """
        dmap_info = {}
        for tup in get_all_moment_tuples():
            if dtype == tup[0]:
                # name of variable and attribute
                dmap_info['flag'] = ""
                if dname is None: dname = dtype.replace(tup[0], tup[1])
                dmap_info['dname'] = dname
                dmap_info['dtype'] = dtype
                dmap_info['order'] = dict_invert_moments[tup[0].lower()][0]
                dmap_info['dunit'] = dict_invert_moments[tup[0].lower()][3]
                break
        return dmap_info

    def remove_map(self, name):
        """Remove map
        """
        respop = self.maps.pop(name, None)

    def _analyse_kwargs_tobuild_maps(self, **kwargs):
        """Analyse kwargs to extract map_kwargs for each
        associated map.

        Args:
            **kwargs:

        Returns
            List of map_kwargs for further processing
        """
        # Getting all the suffixes/names for the Maps
        dict_maps_suffix = ex_suff(kwargs.keys(), list_Map_attr)
        list_map_kwargs = []
        # Attaching all detected Maps
        for suffix_map in dict_maps_suffix.keys():
            keywords_for_map = dict_maps_suffix[suffix_map]
            # Initialise the kwargs for Map
            map_kwargs = {}
            # First go through all the Map specific keyword
            for key in keywords_for_map:
                map_kwargs[key] = kwargs.pop(add_suffix(key, suffix_map, link=""))
            if 'name' not in map_kwargs.keys():
                if suffix_map != "":
                    map_name = suffix_map
                else:
                    map_name = "Map{0:02d}".format(self.nmaps + 1)
            map_kwargs['name'] = map_name

            # Then add the ones for the datamaps
            # First detect all the datamaps keys for this specific Map
            list_Data_attr_up = [add_suffix(dattr, suffix_map, link="")
                                 for dattr in list_Data_attr]
            # And now get the suffixes for these
            dict_dmaps_suffix = ex_suff(kwargs.keys(), list_Data_attr_up)
            for suffix_dmap in dict_dmaps_suffix:
                list_key_dmap = dict_dmaps_suffix[suffix_dmap]
                for key_dmap in list_key_dmap:
                    item = key_dmap.replace(suffix_map, "")
                    map_kwargs[item] = kwargs.pop(key_dmap, None)

            list_map_kwargs.append(map_kwargs)

        return list_map_kwargs

    def add_maps(self, **kwargs):
        """Add a set of maps defined via kwargs
        First by analysing the input kwargs, and then processing
        them one by one to add the maps

        Args:
            **kwargs:
        """
        list_map_kwargs = self._analyse_kwargs_tobuild_maps(**kwargs)
        for map_kwargs in list_map_kwargs:
            self.add_map(**map_kwargs)

    def add_map(self, **kwargs):
        """Attach a new map from kwargs
        It forces the datatype to follow the pre-defined keys
        """

        # If we force predefined types, then go through them
        # And see if they correspond
        if self.force_dtypes:
            if 'dtype' in kwargs.keys():
                dtype = kwargs.get("dtype")
                dname = kwargs.get("dname", None)
                # See if this is within the pre-defined types
                dmap_info = self._extract_dmap_info_from_dtype(dtype, dname)
                if len(dmap_info) > 0:
                    print("WARNING[add_map]: found pre-defined data type {} "
                          "-> forcing unit, comment and coordinate name".format(dtype))
                    for key in dmap_info.keys():
                        kwargs[key] = dmap_info[key]
                else:
                    print("WARNING[add_map]: found a dtype which is not pre-defined".format(
                          dtype))
            else:
                print("ERROR[add_map]: force_dtypes is set ON but...")
                print("ERROR[add_map]: dtype not in kwargs - Aborting")
                return

        newmap = Map(**kwargs)
        self.attach_map(newmap)

    def attach_map(self, newmap):
        newmap.align_axes(self)
        self.maps[newmap.name] = newmap

    def _has_map(self, name):
        if self.nmaps <= 0:
            return False
        else:
            return name in self.maps.keys()

    def _get_datamap(self, map_name, dname=None, order=None):
        """Get the datamap from a given Map
        """
        # Get the Map
        ds = self._get_map(map_name)
        datamap = ds._get_datamap(dname)
        if datamap is  None:
            return ds, None
        else:
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
                        return ref_data_shape
                    print("WARNING: data {} not a numpy array".format(
                          kwarg_name))

        return ref_data_shape

    def _get_map(self, map_name=None, mtype=""):
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
                if mtype in self.maps[map_name].mtype:
                    break
        return self.maps[map_name]

    def _get_profile(self, profile_name=None, ptype=""):
        """
        Args:
            **kwargs:
        Returns:
            Either profile_name if provided, otherwise
            just the first profile name.
        """
        # if name is None, get the first Profile with the right type
        if profile_name is None :
            for profile_name in self.profiles.keys():
                if ptype == self.profiles[profile_name].ptype:
                    break

        # if still none
        if profile_name is None:
            # If profile_name is still None, get an error raised
            print("ERROR[_get_profile]: could not get profile_name, "
                  "even from ptype {}".format(ptype))
            return None

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
            self.profiles[prof_name] = Profile(data=Vc, edata=eVc, R=R,
                                          dname=prof_name, order=1,
                                          ptype='vel', dunit=kms,
                                          dtype=filetype, **kwargs)
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

    def get_radial_profile(self, map_name, dname=None, order=0):
        """Get a radial profile from a given map. If dname
        is None, it will use the first map of order 0.

        Args:
            map_name:
            dname:
            order:

        Returns:

        """
        pass



