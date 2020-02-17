# -*- coding: utf-8 -*-
"""
Module for the data classes:
    Maps are made of X,Y and associated DataMap(s), while
    Profiles are made of R and associated DataProfile(s)
"""

# External modules
import numpy as np
from numpy import deg2rad

# Units
import astropy.units as u

# Float
from .misc_io import add_suffix, default_float, add_err_prefix
from .misc_io import extract_suffixes_from_kwargs
from .misc_io import AttrDict, remove_suffix
from . import check, transform
from . import local_units as lu

default_data_names = ["data", "edata"]
dict_units = {"XY": u.arcsec, "R": u.arcsec}
dict_kwargs_Map = {'comment':"", 'NE_direct':0,
                   'alpha_North':0, 'filled_value':'nan',
                   'method':'linear'}
# List of attributes to use for Maps and DataMaps
list_Data_attr = ["dorder", "dtype", "dname",
                  "data", "edata", "dunit"]

class DataMap(object):
    """Data class representing a specific map

    Attributes
    ----------
    data
    edata
    order
    name
    """
    def __init__(self, data, edata=None, order=0, name=None, flag=None, dtype="", dunit=None):
        """
        Args:
            data: numpy array
                Input datas values.
            edata: numpy array [None]
                Uncertainties for the data.
            order: int [0]
                order of the velocity moment. Can be -1 for 'others' (grid)
            name: str [None]
                Name of the map.
            flag: str [None]
            dtype: str [""]
            dunit: astropy unit [None]
        """
        self.data = data
        self.edata = edata
        self.order = order
        self.name = name
        self.flag = flag
        self.dtype = dtype
        self.dunit = dunit

    def add_transformed_data(self, data, edata=None, suffix=""):
        """Add a transformed map - e.g, deprojected - using a suffix

        Args:
            data:
            edata:
            suffix:

        Returns:

        """
        if suffix == "" or suffix is None:
            print("[add_transformed_data] Failed to add data, suffix is empty")
            return
        new_data_attr = add_suffix("data", suffix)
        setattr(self, new_data_attr, data)
        new_edata_attr = add_err_prefix(new_data_attr)
        setattr(self, new_edata_attr, edata)

    def _reshape_datamap(self, shape):
        """reshape the data
        Args:
            shape:

        """
        self.data = self.data.reshape(shape)
        if self.edata is not None:
            self.edata = self.edata.reshape(shape)

    def deproject_velocities(self, inclin=90.0):
        """Deproject Velocity map and add it
        """

        if self.order != 1:
            print("ERROR: data are not of order 1 [velocities] -- Aborting")
            return
        Vdep, eVdep = transform.deproject_velocities(self.data,
                                                     self.edata,
                                                     inclin)
        self.add_transformed_data(Vdep, eVdep, "dep")

class DataProfile(DataMap):
    def __init__(self, data, edata=None, **kwargs):
        """Create a profile class with some data and errors.

        Args:
            data:
            edata:
            **kwargs: (see DataMap)
                order: int
                name: str
                flag: str
                dtype: ""
        """
        # Using DataMap class attributes
        super().__init__(data=data, edata=edata, **kwargs)

        # Now 1D in case the input is 2D
        if data is not None:
            self.data = self.data.ravel()
        if edata is not None:
            self.edata = self.edata.ravel()

class Map(object):
    """A Map is a set of DataMaps associated with a location grid (X, Y)
    It is used to describe a set of e.g., velocity fields, flux maps, etc
    A grid is associated natively to these DataMaps as well as an orientation
    on the sky.
    If no grid is provided, the grid is set to integer numbers (pixels).

    Attributes
    ----------
    NE_direct: bool [True]
        If True, direct sense for NE meaning East is counter-clockwise
        from North in the map.
    alpha_North: float [0.]
        Angle of the North w.r.t. top. Positive means counter-clockwise.
    X, Y: numpy arrays [None, None]
        Input location grid
    data: numpy array [None]
        Input values
    edata: numpy array [None]
        Uncertainties
    order: int [0]
        Order of the velocity moment. -1 for 'others' (e.g., X, Y)
    """
    def __init__(self, X=None, Y=None, ref_shape=None,
                 name=None, mtype="", comment="", **kwargs):
        """
        Args:
            X: numpy array [None]
                Input X axis location array
            Y: numpy array [None]
                Input Y axis location array
            Xcen, Ycen: float, float
                Centre for the X and Y axes. Default is the centre of the image.
            ref_shape: tuple [None]
                Reference shape for the arrays. If provided, will be used to shape the
                input arrays
            name: str [None]
                Name of the dataset
            comment: str [""]
                Comment attached to the Map

            **kwargs:
                Any of these attributes can be provided with
                a suffix. E.g., "dataCO" will be understood
                as data and with a flag=CO.
                data: array
                edata: array [None]
                    Uncertainty map
                name: str
                    Name of the map
                mtype: str
                    Type of the map
                flag: str
                    Flag for the map
                order: int
                    Order for the datamap
        """
        # Empty dictionary for the moments
        self.dmaps = AttrDict()

        # We assume that by default the units are the default arcsec
        self.XYunit = kwargs.pop("XYunit", dict_units['XY'])

        # Get the list of suffixes which will be used to attach datasets
        dict_data_suffix = extract_suffixes_from_kwargs(kwargs.keys(),
                                                           list_Data_attr)
        suffix_data = None
        for suffix in dict_data_suffix.keys():
            if "data" in dict_data_suffix[suffix]:
                suffix_data = suffix
                data = kwargs.get("data" + suffix_data)
                break

        # First getting the shape of the data
        if ref_shape is not None:
            self.shape = ref_shape
        # Using X if ref_shape is None
        elif X is not None:
            self.shape = X.shape
        # If no X, try with the data
        else:
            if suffix_data is not None:
                self.shape = data.shape
            else:
                print("ERROR: no reference shape is provided "
                      "(via X, data or ref_shape) - Aborting")
                return

        # Initialise the X, Y coordinates
        self.Xcen = kwargs.pop("Xcen", 0.)
        self.Ycen = kwargs.pop("Ycen", 0.)
        # Pixel scale for the X, Y coordinates
        self.pixel_scale = kwargs.pop("pixel_scale", 1.)
        if not self._init_XY(X, Y):
            print("ERROR: X and Y are not compatible - Aborting.")
            return

        # Boolean to say whether E is direct from N (counter-clockwise)
        self.NE_direct = kwargs.pop("NE_direct", True)
        # Angle (degrees) between North and top
        self.alpha_North = kwargs.pop("alpha_north", 0.)
        # Get the matrix for the alignment with North-up
        self.align_xy_NorthEast()

        # Filling value
        self._fill_value = kwargs.pop("fill_value", 'nan')
        # Method
        self._method = kwargs.pop("method", "linear")

        # Comment, name and map type
        self.comment = comment
        self.name = name
        self.mtype = mtype
        self.overwrite = kwargs.pop("overwrite", False)

        self.add_datamaps_from_kwargs(**kwargs)

    def add_datamaps_from_kwargs(self, **kwargs):
        """Process the list of kwargs to extract datamaps
        and attach them
        Attributes
            **kwargs
        """
        dict_data_suffix = extract_suffixes_from_kwargs(kwargs.keys(),
                                                           list_Data_attr)
        # Attaching all detected datamaps
        for suffix in dict_data_suffix.keys():
            key_list = dict_data_suffix[suffix]
            mykwargs = {}
            for key in key_list:
                mykwargs[key] = kwargs.pop(add_suffix(key, suffix, link=""), None)
            if not 'flag' in key_list:
                mykwargs['flag'] = "datamap{0:02d}".format(len(self.dmaps) + 1)
            if not 'dname' in key_list:
                if suffix != "":
                    mykwargs['dname'] = suffix
            self.add_data(**mykwargs)

    def __getattr__(self, name):
        for suffix in default_data_names:
            if name.startswith(suffix):
                for mapname in self.dmaps.keys():
                    if mapname in name:
                        basename = remove_suffix(name, mapname)
                        return getattr(self.dmaps[mapname], basename)

    def __dir__(self, list_names=default_data_names):
        return  super().__dir__() + [add_suffix(attr, map) for item in list_names
                for map in self.dmaps.keys() for attr in self.dmaps[map].__dir__()
                if attr.startswith(item)]

    def _init_XY(self, X, Y):
        """Initialise X and Y

        Args:
            X: numpy array
            Y: numpy array
                Input X, Y grid.
        """
        # Define the grid in case X, Y not yet defined
        # If it is the case, using the reference Map
        if X is None or Y is None:
            # We get the grid in pixel
            print("WARNING: X or Y not provided. Using Pixel XY grid.")
            ref_ind = np.indices(self.shape, dtype=default_float)
            self.X = ref_ind[1] - (self.shape[1] - 1.) / 2.
            self.Y = ref_ind[0] - (self.shape[0] - 1.) / 2.
            # And now convert to default unit
            self._convert_to_xyunit()
        else:
            # if X, Y pre-defined, unit is pre-defined too
            if not check._check_consistency_sizes([X, Y]):
                print("ERROR: errors on sizes of X and Y")
                return False
            # Just removing the centre to get 0,0
            self.X = X - self.Xcen
            self.Y = Y - self.Ycen

        # Making sure the shapes agree
        self.X = self.X.reshape(self.shape)
        self.Y = self.Y.reshape(self.shape)
        return True

    @property
    def eq_pscale(self):
        return u.pixel_scale(self.pixel_scale * self.XYunit / u.pixel)

    @property
    def _get_pixel_scale(self):
        return (1. * u.pixel).to(self.XYunit, equivalencies=self.eq_pscale).value

    def _convert_to_xyunit(self):
        """Convert XYunit into the default one
        a priori arcseconds.
        """
        self.X *= self.xyunit_per_pixel
        self.Y *= self.xyunit_per_pixel
        # Update the unit
        self.XYunit = dict_units['XY']

    @property
    def xyunit_per_pixel(self):
        return (1. * self.XYunit).to(dict_units['XY'],
                                 self.eq_pscale).value

    def _get_datamap(self, name=None, order=None):
        if name is None:
            if order is None:
                # then just get the first map
                name = list(self.dmaps.keys())[0]
            else:
                # Then get the first map of right order
                for key in self.dmaps.keys():
                    if self.dmaps[key].order == order:
                        name = key
                        break

        if self._has_datamap(name):
            return self.dmaps[name]
        else:
            print("No such datamap {} in this Map".format(name))
            return None

    def _has_datamap(self, name):
        return name in self.dmaps.keys()

    def _regrid_xydatamaps(self):
        if not check._check_ifnD([self.X], ndim=2):
            print("WARNING: regridding X, Y and datamaps into 2D arrays")
            newextent, newX, newY = transform.regrid_XY(self.X, self.Y)
            for name in self.dmaps.keys():
                self.dmaps[name].data = transform.regrid_Z(self.X, self.Y,
                                                           self.dmaps[name].data,
                                                           newX, newY,
                                                           fill_value=self._fill_value,
                                                           method=self._method)
                self.dmaps[name].edata = transform.regrid_Z(self.X, self.Y,
                                                            self.dmaps[name].edata,
                                                            newX, newY,
                                                            fill_value=self._fill_value,
                                                            method=self._method)
            # Finally getting the new X and Y
            self.X, self.Y = newX, newY
            self.shape = self.X.shape

    def _reshape_datamaps(self):
        """Reshape all datamaps following X,Y shape
        """
        for name in self.dmap.keys():
            self.dmaps[name].reshape(self.shape)

    def attach_datamap(self, datamap):
        """Attach a DataMap to this Map

        Args:
            datamap: a DataMap
        """
        if self._check_datamap(datamap):
            datamap._reshape_datamap(self.shape)
            self.dmaps[datamap.name] = datamap
        else:
            print("WARNING[attach_datamap]: could not attach datamap")

    def add_data(self, data=None, order=0, edata=None,
                    dname=None, flag=None, dtype=None,
                    dunit=None, **kwargs):
        """Add a new DataMap to the present Map. Will check if
        grid is compatible.

        Args:
            data: 2d array
            order: int
            edata: 2d array
            dname: str
            dtype: str
            flag: str
            dunit: astropy unit
        """
        if data is None:
            print("WARNING[attach_data]: cannot attach data: it is 'None'")
            return

        overwrite = kwargs.pop("overwrite", self.overwrite)

        # Input name to define the data. If none, define using the counter
        if dname is None:
            dname = "map{0:02d}".format(len(self.dmaps)+1)

        if self._has_datamap(dname) and not overwrite:
            print("WARNING[attach_data]: data map {} already exists "
                  "- Aborting".format(dname))
            print("WARNING[attach_data]: use overwrite option to force.")
            return

        self.attach_datamap(DataMap(data, edata, order,
                                    dname, flag, dtype,
                                    dunit))

    def _check_datamap(self, datamap):
        """Check consistency of data
        """
        # Main loop on the names of the dmaps
        arrays_to_check = [datamap.data.ravel()]
        if datamap.edata is not None:
            arrays_to_check.append(datamap.edata.ravel())

        # First checking that the data are arrays
        if not check._check_ifarrays(arrays_to_check):
            print("ERROR[check_datamap]: input maps not all arrays")
            return False

        # Then checking that they are consistent with X, Y
        arrays_to_check.insert(0, self.X.ravel())
        if not check._check_consistency_sizes(arrays_to_check):
            print("ERROR[check_datamap]: input datamap does not "
                  "have the same size than input grid (X, Y)")
            return False

        return True

    def align_axes(self, galaxy):
        """Align all axes using X and Y as input
        """
        self.align_xy_lineofnodes(galaxy)
        self.align_xy_bar(galaxy)
        self.align_xy_deproj_bar(galaxy)

    @property
    def XY_extent(self):
        return [np.min(self.X), np.max(self.X),
                np.min(self.Y), np.max(self.Y)]

    @property
    def _R(self):
        return np.sqrt(self.X**2 + self.Y**2)

    # Setting up NE direct or not
    @property
    def NE_direct(self) :
        return self.__NE_direct

    @NE_direct.setter
    def NE_direct(self, NE_direct) :
        self.__NE_direct = NE_direct
        self._mat_direct = np.where(NE_direct,
                                    transform.set_stretchmatrix(),
                                    transform.set_reverseXmatrix())
    # Setting up North-East to the top
    @property
    def alpha_North(self) :
        return self.__alpha_North

    @alpha_North.setter
    def alpha_North(self, alpha_North) :
        """Initialise the parameters in the disc structure for alpha_North angles
        in degrees and radian, as well as the associated transformation matrix

        Input
        -----
        alpha_North: angle in degrees for the PA of the North direction
        """
        self.__alpha_North = alpha_North
        self.__alpha_North_rad = deg2rad(alpha_North)
        self._mat_NE = self._mat_direct * transform.set_rotmatrix(self.__alpha_North_rad)

    def _get_angle_from_PA(self, PA):
        """Provide a way to get the angle within the original
        frame of a certain axis with a given PA
        Args:
            PA: float
                PA of axis with respect to North

        Returns:
            The angle in the original frame
        """
        return PA + self.alpha_North * np.where(self.NE_direct, 1., -1.)

    def align_xy_NorthEast(self) :
        """Get North to the top and East on the left
        """
        self.X_NE, self.Y_NE = self.rotate(matrix=self._mat_NE)

    def align_xy_lineofnodes(self, galaxy) :
        """Set the Line of Nodes (defined by its Position Angle, angle from the North
        going counter-clockwise) as the positive X axis
        """
        self._mat_lon_NE = galaxy._mat_lon * self._mat_NE
        self.X_lon, self.Y_lon = self.rotate(matrix=self._mat_lon_NE)

    def deproject(self, galaxy):
        """Deproject X,Y around the line of nodes using the inclination
        """
        self.X_londep, self.Y_londep = self.rotate(matrix=galaxy._mat_inc,
                                             X=self.X_lon, Y=self.Y_lon)

    def align_xy_bar(self, galaxy) :
        """Set the bar (defined by its Position Angle, angle from the North
        going counter-clockwise) as the positive X axis
        """
        self.X_bar, self.Y_bar = self.rotate(matrix=galaxy._mat_bar * self._mat_NE)

    def align_xy_deproj_bar(self, galaxy) :
        """Set the bar (defined by its Position Angle, angle from the North
        going counter-clockwise) as the positive X axis after deprojection
        """
        self._mat_deproj_bar = galaxy._mat_bardep * galaxy._mat_inc * galaxy._mat_lon * self._mat_NE
        self.X_bardep, self.Y_bardep = self.rotate(matrix=self._mat_deproj_bar)

        ## Mirroring the coordinates
        self.X_mirror, self.Y_mirror = self.rotate(matrix=np.linalg.inv(self._mat_deproj_bar),
                                                   X=self.X_bardep, Y=-self.Y_bardep)

    def rotate(self, **kwargs):
        """Uses the rotate function from transform.py with a default
        X,Y set of arrays using self.X and self.Y

        Parameters
        ----------
        **kwargs: set of arguments, see transform.rotate
            Includes X, Y, matrix

        Returns:
        The rotated arrays Xrot, Yrot
        """
        X = kwargs.pop("X", self.X)
        Y = kwargs.pop("Y", self.Y)
        return transform.rotate_vectors(X, Y, **kwargs)

    def deproject_velocities(self, map_name, inclin=90.0):
        """Deproject Velocity map if it exists

        Parameters
        ----------
        map_name: str
            Name of the map to deproject
        inclin: float [90]
            Inclination in degrees
        """

        if map_name in self.dmaps:
            self.dmaps[map_name].deproject_velocities(inclin=inclin)
        else:
            print("ERROR: no such data name in this Map")

class Profile(object):
    """A Profile is a set of DataProfiles associated via the same R profile.
    It is used to describe radial dprofiles e.g., rotation curves.

    Attributes
    ----------
    Rin: numpy array [None]
        Input location radii
    data: numpy array [None]
        Input values
    edata: numpy array [None]
        Uncertainties
    order: int [0]
        Order of the velocity moment. -1 for 'others' (e.g., X, Y)
    """
    def __init__(self, R=None, ref_size=None,
                 name=None, ptype="", **kwargs):
        """
        Args:
            Rin: numpy array [None]
                Input R axis location array
            name: str [None]
                Name of the dataset1d

            **kwargs:
                data: array
                edata: array [None]
                    Uncertainty profile
                name: str
                    Name of the profile
                ptype: str
                    Type of the profile
                flag: str
                    Flag for the profile
                order: int
                    Order for the profile
                comment: str [""]
                    Comment attached to the Profile
        """
        # Empty dictionary for the moments
        self.dprofiles = AttrDict()

        # See if a dataprofile is provided
        self.Runit = kwargs.pop("Runit", dict_units['R'])
        self.pixel_scale = kwargs.pop("pixel_scale", 1.)

        # Get the list of suffixes which will be used to attach datasets
        dict_data_suffix = extract_suffixes_from_kwargs(kwargs.keys(),
                                                           list_Data_attr)
        suffix_data = None
        for suffix in dict_data_suffix.keys():
            if "data" in dict_data_suffix[suffix]:
                suffix_data = suffix
                data = kwargs.get("data" + suffix_data)
                break

        # First getting the shape of the data
        if ref_size is not None:
            self.size = ref_size
        elif R is not None:
            self.size = R.size
        else:
            if suffix_data is not None:
                self.size = data.size
            else:
                print("ERROR: no reference shape is provided "
                      "(via R, data or ref_size) - Aborting")
                return

        # New step in R when provided
        Rfinestep = kwargs.pop("Rfinestep", 0)
        self._init_R(R)

        # Filling value
        self._fill_value = kwargs.pop("fill_value", 'nan')
        # Method
        self._method = kwargs.pop("method", "linear")

        # Comment for Profile
        self.comment = kwargs.pop("comment", "")
        # Name of Profile
        self.name = name
        self.ptype = ptype
        self.overwrite = kwargs.pop("overwrite", False)

        self.add_dataprofiles_from_kwargs(**kwargs)

        if Rfinestep > 0:
            self.interpolate(newstep=Rfinestep)

    def add_dataprofiles_from_kwargs(self, **kwargs):
        """Process the list of kwargs to extract dataprofiles
        and attach them
        Attributes
            **kwargs
        """
        dict_data_suffix = extract_suffixes_from_kwargs(kwargs.keys(),
                                                           list_Data_attr)
        # Attaching all detected data profiles
        for suffix in dict_data_suffix.keys():
            key_list = dict_data_suffix[suffix]
            mykwargs = {}
            for key in key_list:
                mykwargs[key] = kwargs.pop(add_suffix(key, suffix, link=""), None)
            if not 'flag' in key_list:
                mykwargs['flag'] = "dataprof{0:02d}".format(len(self.dprofiles) + 1)
            if not 'dname' in key_list:
                if suffix != "":
                    mykwargs['dname'] = suffix
            self.add_data(**mykwargs)

    @property
    def eq_pscale(self):
        return u.pixel_scale(self.pixel_scale * self.Runit / u.pixel)

    @property
    def scale_runit(self):
        return (1. * self.Runit).to(dict_units['R'],
                                     self.eq_pscale).value

    def _convert_to_runit(self):
        """Convert Runit into the default one
        a priori arcseconds.
        """
        self.R *= self.scale_runit
        # Update the unit
        self.Runit = dict_units['R']

    def _init_R(self, R):
        """Initialise Rin

        Args:
            Rin: numpy array
            nameR:
        """
        # Define the grid in case Rin
        # If it is the case, using the reference profile
        if R is None:
            self.R = np.arange(self.size, dtype=default_float)
        else:
            self.R = R.ravel()
        self._convert_to_runit()

    def _has_profile(self, name):
        return name in self.dprofiles.keys()

    def attach_dataprofile(self, dataprofile):
        """Attach a DataProfile to this Profile

        Args:
            dataprofile: DataProfile to attach
        """
        if self._check_dprofiles(dataprofile):
            self.dprofiles[dataprofile.name] = dataprofile

    def add_data(self, data, order=0, edata=None,
                       dname=None, flag=None, dtype=None,
                       dunit=None, overwrite=False):
        """Attach a new Profile to the present Set.

        Args:
            data: 1d array
            order: int
            edata: 1d array
            dname: str
            dtype: str
            flag: str
            dunit: astropy unit

        """
        if data is None:
            return

        # Input name to define the data. If none, define using the counter
        if dname is None:
            dname = "prof{0:02d}".format(len(self.dprofiles)+1)

        if self._has_profile(dname) and not overwrite:
            print("WARNING[add_data]: data profile {} already exists "
                  "- Aborting".format(dname))
            print("WARNING[add_data]: use overwrite option to force.")
            return

        self.attach_dataprofile(DataProfile(data, edata, order=order,
                                            name=dname, flag=flag,
                                            dtype=dtype, dunit=dunit))

    def __getattr__(self, name):
        for suffix in default_data_names:
            if name.startswith(suffix):
                for profname in self.dprofiles.keys():
                    if profname in name:
                        basename = remove_suffix(name, profname)
                        return getattr(self.dprofiles[profname], basename)

    def __dir__(self, list_names=default_data_names):
        return  super().__dir__() + [add_suffix(attr, prof) for item in list_names
                for prof in self.dprofiles.keys() for attr in self.dprofiles[prof].__dir__()
                if attr.startswith(item)]

    def _check_dprofiles(self, dataprofile):
        """Check consistency of dataprofile
        by comparing with self.Rin

        Args
            dataprofile: DataProfile
        """
        # Putting everything in 1D
        ref_array = self.R

        data = dataprofile.data
        edata = dataprofile.edata
        arrays_to_check = [data.ravel()]
        if edata is not None:
            arrays_to_check.append(edata.ravel())

        # Checking if the data are 1D arrays
        if not check._check_ifarrays(arrays_to_check):
            print("ERROR: input profile not all arrays")
            return False

        # Check that they all have the same size
        arrays_to_check.insert(0, ref_array)
        if not check._check_consistency_sizes(arrays_to_check):
            print("ERROR: input profile does not the same size "
                  "than input radial grid (R)")
            return False

        return True

    def interpolate(self, name, step=1.0, suffix="fine", overwrite=False):
        """Provide interpolated profile

        Args:
            stepR: float [1.0]
            suffix: str [""]
            overwrite: bool [False]

        Returns:

        """
        if step <= 0:
            print("ERROR[interpolate]: new step is <= 0 - Aborting")
            return

        # Getting the data
        if not self._has_profile(name):
            print("ERROR[interpolate]: no such map with name {}".format(name))
            return

        if hasattr(self.dprofiles[name], add_suffix("R", suffix)):
            if overwrite:
                print("WARNING: overwriting existing interpolated profile")
            else:
                print("ERROR[interpolate]: interpolated profile exists. "
                      "Use 'overwrite' to update.")
                return

        Rfine, dfine, edfine = transform.interpolate_profile(self.R,
                                                             self.dprofiles[name].data,
                                                             self.dprofiles[name].edata,
                                                             step=step)
        setattr(self.dprofiles[name], add_suffix("R", suffix), Rfine)
        setattr(self.dprofiles[name], add_suffix("data", suffix), dfine)
        setattr(self.dprofiles[name], add_suffix("edata", suffix), edfine)

class Slicing(object):
    """Provides a way to slice a 2D field. This class just
    computes the slits positions for further usage.
    """
    def __init__(self, yextent=[-10.,10.], yin=None, slit_width=1.0, nslits=None):
        """Initialise the Slice by computing the number of slits and
        their positions (defined by the axis 'y').

        Args:
            yextent: list of 2 floats
                [ymin, ymax]
            yin: numpy array
                input y position
            slit_width: float
                Width of the slit
            nslits: int
                Number of slits. This is optional if a range or input yin
                is provided.
        """

        # First deriving the range. Priority is on yin
        if yin is not None:
            yextent = [np.min(yin), np.max(yin)]
        # First deriving the number of slits prioritising nslits

        Dy = np.abs(yextent[1] - yextent[0])
        if nslits is None:
            self.nslits = np.int(Dy / slit_width + 1.0)
            ye2 = (Dy - self.nslits * slit_width) / 2.
            # Adding left-over on both sides equally
            yextent = [yextent[0] - ye2, yextent[1] + ye2]
        else:
            self.nslits = nslits
            slit_width = Dy / self.nslits

        self.width = slit_width
        sw2 = slit_width / 2.
        self.ycentres = np.linspace(yextent[0] + sw2, yextent[1] - sw2, self.nslits)
        self.yedges = np.linspace(yextent[0], yextent[1], self.nslits+1)
        self.yiter = np.arange(self.nslits)

        @property
        def yextent(self):
            return [self.yedges[0], self.yedges[-1]]

        @property
        def slice_width(self):
            return np.abs(self.yedges[-1] - self.yedges[0])
