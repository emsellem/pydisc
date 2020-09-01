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
from os.path import join as joinpath

# local modules
from .galaxy import Galaxy
from .disc_data import Map, Profile, match_datamaps
from .misc_io import AttrDict, read_vc_file
from .maps_grammar import analyse_maps_kwargs, list_Map_attr, list_Profile_attr
from .maps_grammar import extract_suff_from_keywords, default_data_separator
from . import local_units as lu

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
                PAnodes
                PAbar
        """
        self.verbose = kwargs.pop("verbose", False)

        # initialise Maps if any
        self._reset_maps()
        self._reset_profiles()

        # Using galaxy class attributes
        super().__init__(**kwargs)

        self._force_dtype = kwargs.pop("force_dtype", True)

        # Adding the maps
        self.add_maps(**kwargs)

        # Adding the maps
        self.add_profiles(**kwargs)

        # init slicing
        self._reset_slicing()

    def __getattr__(self, item):
        names = item.split(default_data_separator)
        if len(names) ==2 and names[0] in self.maps.keys():
            if names[1] in self.maps[names[0]].dmaps.keys():
                return self.maps[names[0]].dmaps[names[1]]
        raise AttributeError("'GalacticDisc' object has no attribute {}".format(item))

    def _decode_prof_name(self, name):
        list_prof_names = list(self.profiles.keys())
        dict_profs = extract_suff_from_keywords([name], list_prof_names, separator=default_data_separator)
        if len(dict_profs) == 0:
            return None, None

        dname = list(dict_profs.keys())[0]
        pname = dict_profs[dname][0]
        if dname not in self.profiles[pname].dprofiles:
            return pname, None
        else:
            return pname, dname

    def _decode_map_name(self, name):
        list_map_names = list(self.maps.keys())
        dict_maps = extract_suff_from_keywords([name], list_map_names, separator=default_data_separator)
        if len(dict_maps) == 0:
            return None, None

        dname = list(dict_maps.keys())[0]
        mname = dict_maps[dname][0]
        if dname not in self.maps[mname].dmaps:
            return mname, None
        else:
            return mname, dname

    @property
    def _list_valid_dtypes(self):
        return get_all_moment_types()

    def _get_slicing(self, slicing_name):
        if slicing_name not in self.slicings:
            if bool(self.slicings):
                return self.slicings[list(self.slicings.keys())[0]]
            else:
                return None
        else:
            return self.slicings[slicing_name]

    def _reset_slicing(self):
        """Initialise the slice dictionary if needed
        """
        if not hasattr(self, "slicings"):
            self.slicings = AttrDict()

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

    def remove_map(self, name):
        """Remove map
        """
        respop = self.maps.pop(name, None)

    def add_maps(self, **kwargs):
        """Add a set of maps defined via kwargs
        First by analysing the input kwargs, and then processing
        them one by one to add the data

        Args:
            **kwargs:
        """
        list_map_kwargs = analyse_maps_kwargs(reference_list=list_Map_attr,
                                              **kwargs)
        for name_map, map_kwargs in list_map_kwargs.items():
            map_kwargs["force_dtype"] = self._force_dtype
            if 'mname' not in map_kwargs:
                if name_map == "":
                    name_map = "Map{0:02d}".format(self.nmaps+1)
                map_kwargs['mname'] = name_map
            try:
                self.attach_map(Map(**map_kwargs))
            except ValueError as err:
                print(repr(err))

    def add_profiles(self, **kwargs):
        """Add a set of profiles defined via kwargs
        First by analysing the input kwargs, and then processing
        them one by one to add the data

        Args:
            **kwargs:
        """
        list_prof_kwargs = analyse_maps_kwargs(reference_list=list_Profile_attr,
                                                  **kwargs)
        for name_prof, prof_kwargs in list_prof_kwargs.items():
            prof_kwargs["force_dtype"] = self._force_dtype
            if 'pname' not in prof_kwargs:
                if name_prof == "":
                    name_prof = "Prof{0:02d}".format(self.nprofs+1)
                prof_kwargs['pname'] = name_prof
            try:
                self.attach_profile(Profile(**prof_kwargs))
            except ValueError as err:
                print(repr(err))

    def attach_map(self, newmap):
        """Attaching the map newmap

        Args:
            newmap (Map):

        Returns:

        """
        if newmap is None:
            print("ERROR[attach_map]: cannot attach map (None) - Aborting")
            return

        print("INFO: attaching map {0}".format(newmap.mname))
        newmap.align_axes(self)
        self.maps[newmap.mname] = newmap

    def attach_profile(self, newprof):
        """Attaching the profile newprof

        Args:
            newprof (Profile):

        Returns:

        """
        if newprof is None:
            print("ERROR[attach_prof]: cannot attach profile (None) - Aborting")
            return

        print("INFO: attaching profile {}".format(newprof.pname))
        self.profiles[newprof.pname] = newprof

    def _has_map_data(self, mname, dname, **kwargs):
        if not hasattr(self, 'maps'):
            return False
        if self._has_map(mname):
            if self.maps[mname]._has_datamap(dname):
                found = True
                dmap = self.maps[mname].dmaps[dname]
                for key in kwargs:
                    if hasattr(dmap, key):
                        if kwargs[key] != getattr(dmap, key):
                            found = False
                    else:
                        found = False
                return found

        else:
            return False

    def _has_profile_data(self, pname, dname, **kwargs):
        if not hasattr(self, 'profiles'):
            return False
        if self._has_profile(pname):
            if self.profiles[pname]._has_dataprofile(dname):
                found = True
                dprof = self.profiles[pname].dprofiles[dname]
                for key in kwargs:
                    if hasattr(dprof, key):
                        if kwargs[key] != getattr(dprof, key):
                            found = False
                    else:
                        found = False
                return found
        else:
            return False

    def _has_map(self, mname):
        """Test if it has this map

        Args:
            mname (str): name of the map

        Returns:
            Bool
        """
        if self.nmaps <= 0:
            return False
        else:
            return mname in self.maps.keys()

    def _has_profile(self, pname):
        if self.nprofiles <= 0:
            return False
        else:
            return pname in self.profiles.keys()

    def _get_dataprofile(self, pname, dname=None, order=None):
        """Get the dataprofile and profile from the names
        CHeck order if provided

        Args:
            pname (str): name of the profile
            dname (str): name of the dataprofile
            order (int): order to check [None = ignored]

        Returns:
            name of the profile and dataprofile
        """
        # Get the Profile and DataProfile
        # Test if order is correct only if provided
        ds = self._get_map(pname)
        if ds is not None:
            dataprof = ds._get_dataprofile(dname, order)
        else:
            dataprof = None

        return ds, dataprof

    def _get_datamap(self, mname, dname=None, order=None):
        """Get the datamap and map from the names
        CHeck order if provided

        Args:
            mname (str): name of the map
            dname (str): name of the datamap
            order (int): order to check [None = ignored]

        Returns:
            name of the map and datamap
        """
        # Get the Map and Datamap
        # Test if order is correct only if provided
        ds = self._get_map(mname)
        if ds is not None:
            datamap = ds._get_datamap(dname, order)
        else:
            datamap = None

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

    def _get_map(self, mname=None, mtype=""):
        """
        Args:
            **kwargs:
        Returns:
            Either Map if provided, otherwise
            just the first Map name.
        """
        # if name is None, get the first Map
        if mname is None:
            list_names = [name for name, map in self.maps.items()
                          if mtype==map.mtype]
            if len(list_names) > 0:
                mname = list_names[0]
            else:
                return None

        if mname not in self.maps.keys():
            return None
        else:
            return self.maps[mname]

    def _get_profile(self, pname=None, ptype=""):
        """
        Args:
            **kwargs:
        Returns:
            Either profile_name if provided, otherwise
            just the first profile name.
        """
        # if name is None, get the first Profile with the right type
        if pname is None :
            for pname in self.profiles.keys():
                if ptype == self.profiles[pname].ptype:
                    break

        # if still none
        if pname is None:
            # If profile_name is still None, get an error raised
            print("ERROR[_get_profile]: could not get pname, "
                  "even from ptype {}".format(ptype))
            return None

        return self.profiles[pname]

    def add_vprofile(self, filename=None, filetype="ROTCUR", folder="",
                     vprof_name=None, **kwargs):
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
            if vprof_name is None:
                vprof_name = "Vprof_{:02d}".format(self.nprofiles + 1)
            dname = "vel01"
            new_profile = Profile(pname=vprof_name, data=Vc, edata=eVc, R=R,
                                  dname=dname, order=1, ptype=filetype,
                                  dunit=lu.kms, dtype='vel', **kwargs)
            self.profiles[vprof_name] = new_profile
            if self.verbose:
                print("Vc file successfully read")
            fullname = new_profile._fullname(dname)
        else:
            print("ERROR status {}".format(status))
            fullname = None

        return fullname

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

    def match_datamaps(self, mname1, mname2,
                       dname1=None, dname2=None,
                       odname1=None, odname2=None):
        """Align two datamaps from two maps

        Args:
            mname1 (str): name of the first map
            mname2 (str): name of the second map. If None, will use the first
            dname1 (str): name of the first datamap
            dname2 (str): name of the second datamap
            odname1 (str): name of the output first datamap
            odname2 (str): name of the output second datamap

        """
        # If map2_name is None, use None, which will make the process use the
        # same map (map1)
        map1 = self._get_map(mname1)
        map2 = self._get_map(mname2)

        # Call the transform module function
        newmap = match_datamaps(map1, map2, dname1, dname2, odname1,
                                odname2, PAnodes=self.PAnodes)
        if newmap is None:
            print("[match_datamaps] Failed matching - Aborting")
            return None

        self.attach_map(newmap)

        # Deprojecting this one
        self.deproject_nodes(newmap.mname)
        return newmap.mname
