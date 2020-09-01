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

# various modules from local lib
from .maps_grammar import (analyse_maps_kwargs, analyse_data_kwargs,
                           default_data_separator, list_Data_attr,
                           get_dmap_info_from_dtype, _add_density_suffix,
                           _is_flag_density, remap_suffix, _add_density_prefix)

from .misc_io import (add_suffix, add_prefix, default_float, add_err_prefix,
                      AttrDict, remove_suffix, guess_stepxy, get_extent,
                      cover_linspace)

from . import check, transform

default_data_names = ["data", "edata"]
dict_units = {"XY": u.arcsec, "R": u.arcsec}

class DataMap(object):
    """Data class representing a specific map

    Attributes
    ----------
    data
    edata
    order
    dname
    dunit
    flag
    """
    def __init__(self, **kwargs):
        """
        Args:
            data: numpy array
                Input datas values.
            edata: numpy array [None]
                Uncertainties for the data.
            order: int [0]
                order of the velocity moment. Can be -1 for 'others' (grid)
            dname: str [None]
                Name of the datamap
            flag: str [None]
            dtype: str [""]
            dunit: astropy unit [None]
        """
        # Using the key list in list_Data_attr to set the attributes
        for keyword in list_Data_attr:
            setattr(self, keyword, kwargs.pop(keyword, None))

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
    def __init__(self, **kwargs):
        """Create a profile class with some data and errors.

        Args:
            data:
            edata:
            **kwargs: (see DataMap)
                order: int
                mname: str
                flag: str
                dtype: ""
        """
        # Using DataMap class attributes
        super().__init__(**kwargs)

        # Now 1D in case the input is 2D
        if self.data is not None:
            self.data = self.data.ravel()
        if self.edata is not None:
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
    def __init__(self, X=None, Y=None, mname=None, mtype="", comment="", **kwargs):
        """
        Args:
            X: numpy array [None]
                Input X axis location array
            Y: numpy array [None]
                Input Y axis location array
            Xcen, Ycen: float, float
                Centre for the X and Y axes. Default is the centre of the image.
            mname: str [None]
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
                mname: str
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

        # ----------- First the default attributes ----------------
        # We assume that by default the units are the default arcsec
        self.XYunit = kwargs.pop("XYunit", dict_units['XY'])

        # Filling value
        self._fill_value = kwargs.pop("fill_value", 'nan')
        # Method
        self._method = kwargs.pop("method", "linear")

        # Comment, mname and map type
        self.comment = comment
        self.mname = mname
        self.mtype = mtype
        self.overwrite = kwargs.pop("overwrite", False)
        self._force_dtype = kwargs.pop("force_dtype", False)

        # ----------------- End of default attributes -------------

        # ----------- Analyse the kwargs ----------------------
        dict_dmaps, map_kwargs = analyse_data_kwargs(**kwargs)

        # ------------ X and Y for the map ---------------
        # First getting the shape of the data
        # Using X first
        if X is not None:
            self.shape = X.shape
        # If no X, try with the data
        else:
            # Look for the first data which exists
            data = None
            for name_dmap in dict_dmaps.keys():
                if "data" in dict_dmaps[name_dmap]:
                    data = dict_dmaps[name_dmap]['data']
                    break

            if check._check_ifarrays([data]):
                    self.shape = data.shape
            else:
                raise ValueError("No reference shape is provided "
                      "for map '{}'(via X, data) - Ignoring this map".format(
                       mname))
                return

        # Initialise the X, Y coordinates
        self.Xcen = kwargs.pop("Xcen", 0.)
        self.Ycen = kwargs.pop("Ycen", 0.)
        # Pixel scale for the X, Y coordinates
        self.pixel_scale = kwargs.pop("pixel_scale", 1.)
        if not self._init_XY(X, Y):
            raise ValueError("ERROR: X and Y are not compatible")
            return

        # Boolean to say whether E is direct from N (counter-clockwise)
        self.NE_direct = kwargs.pop("NE_direct", True)
        # Angle (degrees) between North and top
        self.alpha_North = kwargs.pop("alpha_north", 0.)
        # Get the matrix for the alignment with North-up
        self.align_xy_NorthEast()

        # --- Attaching each data map in turn -----------
        # Add each datamap one by one
        for dmap_name, dmap_kwargs in dict_dmaps.items():
            if 'dname' not in dmap_kwargs:
                dmap_kwargs['dname'] = dmap_name

            # Test if we need to force dtype
            self.add_datamap(**dmap_kwargs)

    def __getattr__(self, mname):
        for suffix in default_data_names:
            if mname.startswith(suffix):
                for mapname in self.dmaps.keys():
                    if mapname in mname:
                        basename = remove_suffix(mname, mapname,
                                                 separator=default_data_separator)
                        return getattr(self.dmaps[mapname], basename)
        raise AttributeError("'Map' object has no attribute {}".format(mname))

    def __dir__(self):
        return  super().__dir__() + [add_suffix(attr, map) for item in ['data', 'edata']
                for map in self.dmaps.keys() for attr in self.dmaps[map].__dir__()
                if attr.startswith(item)]

    @property
    def ndatamaps(self):
        return len(self.dmaps)

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
            self.X = ref_ind[1] - self.Xcen
            self.Y = ref_ind[0] - self.Ycen
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

    def add_datamap(self, **kwargs):
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
        # Input dname to define the data. If none, define using the counter
        dname = kwargs.pop("dname", None)
        if dname is None or dname=="":
            dname = "{0}{1:02d}".format(self.mname, self.ndatamaps+1)

        data = kwargs.pop("data", None)
        if data is None:
            print("WARNING[attach_data/Map]: cannot attach data: "
                  "it is 'None' (dname is {}) - Ignoring".format(dname))
            return

        if not check._check_ifarrays([data]):
            print("WARNING[attach_data/Map]: these date are not an array"
                  "(dname is {}) - Ignoring".format(dname))
            return

        overwrite = kwargs.pop("overwrite", self.overwrite)

        if self._has_datamap(dname) and not overwrite:
            print("WARNING[attach_data]: data map {} already exists "
                  "- Aborting".format(dname))
            print("WARNING[attach_data]: use overwrite option to force.")
            return

        # Check if we wish to force the dtype / dunit
        force_dtype = kwargs.pop("force_dtype", self._force_dtype)
        if force_dtype:
            dtype = kwargs.get("dtype", None)
            dmap_info = get_dmap_info_from_dtype(dtype, dname)
            # Transfer
            for key, value in dmap_info.items():
                kwargs[key] = value

        self.attach_datamap(DataMap(data=data, dname=dname, **kwargs))

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
        self.X *= self.xyunit_per_pixel * self.pixel_scale
        self.Y *= self.xyunit_per_pixel * self.pixel_scale
        # Update the unit
        self.pixel_scale = 1.0
        self.XYunit = dict_units['XY']

    @property
    def xyunit_per_pixel(self):
        return (1. * self.XYunit).to(dict_units['XY'],
                                 self.eq_pscale).value

    def _get_datamap(self, dname=None, order=None):
        """Get the datamap if it exists, and
        check the order

        Args:
            dname:
            order:

        Returns:

        """
        if dname is None:
            if order is None:
                # then just get the first map
                dname = list(self.dmaps.keys())[0]
            else:
                # Then get the first map of right order
                for key in self.dmaps.keys():
                    if self.dmaps[key].order == order:
                        dname = key
                        break

        if self._has_datamap(dname):
            return self.dmaps[dname]
        else:
            print("No such datamap {} in this Map".format(dname))
            return None

    def _fullname(self, dname):
        return add_suffix(self.mname, dname, separator=default_data_separator)

    def _has_datamap(self, dname):
        return dname in self.dmaps.keys()

    def _regrid_xydatamaps(self):
        if not check._check_ifnD([self.X], ndim=2):
            print("WARNING: regridding X, Y and datamaps into 2D arrays")
            newextent, newX, newY = transform.regrid_XY(self.X, self.Y)
            for dname in self.dmaps.keys():
                self.dmaps[dname].data = transform.regrid_Z(self.X, self.Y,
                                                           self.dmaps[dname].data,
                                                           newX, newY,
                                                           fill_value=self._fill_value,
                                                           method=self._method)
                self.dmaps[dname].edata = transform.regrid_Z(self.X, self.Y,
                                                            self.dmaps[dname].edata,
                                                            newX, newY,
                                                            fill_value=self._fill_value,
                                                            method=self._method)
            # Finally getting the new X and Y
            self.X, self.Y = newX, newY
            self.shape = self.X.shape

    def _reshape_datamaps(self):
        """Reshape all datamaps following X,Y shape
        """
        for dname in self.dmap.keys():
            self.dmaps[dname].reshape(self.shape)

    def attach_datamap(self, datamap):
        """Attach a DataMap to this Map

        Args:
            datamap: a DataMap
        """
        if self._check_datamap(datamap):
            datamap._reshape_datamap(self.shape)
            self.dmaps[datamap.dname] = datamap
        else:
            print("WARNING[attach_datamap]: could not attach datamap")

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
        self._mat_NE = self._mat_direct @ transform.set_rotmatrix(self.__alpha_North_rad)

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
        self._mat_lon_NE = galaxy._mat_lon.dot(self._mat_NE)
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
        self.X_bar, self.Y_bar = self.rotate(matrix=galaxy._mat_bar @ self._mat_NE)

    def align_xy_deproj_bar(self, galaxy) :
        """Set the bar (defined by its Position Angle, angle from the North
        going counter-clockwise) as the positive X axis after deprojection
        """
        self._mat_deproj_bar = galaxy._mat_bardep @ galaxy._mat_inc @ galaxy._mat_lon @ self._mat_NE
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

    def intmap_to_densitymap(self, dname, galaxy):
        """Change intensity into density quantity
        by dividing by the XYunit**2

        Args:
            dname (str): name of the datamap
            galaxy (Galaxy):

        Does:
            attach a new map with the normalisation

        """
        if not self._has_datamap(dname):
            return

        dmap = self.dmaps[dname]
        # Test if the map is a density one using the type
        if not _is_flag_density(dmap.flag):
            scalepc2 = galaxy.pc_per_xyunit(self.XYunit) ** 2
            newdata = dmap.data / scalepc2
            if dmap.edata is not None:
                newedata = dmap.edata / scalepc2
            else:
                newedata = None
            newdtype = _add_density_suffix(dmap.dtype)
            newdunit = dmap.dunit / self.XYunit**2
            newflag = _add_density_prefix(dmap.flag)
            newdname = _add_density_prefix(dmap.dname)
            self.add_datamap(dname=newdname, data=newdata, edata=newedata,
                             dunit=newdunit, dtype=newdtype,
                             flag=newflag, order=dmap.order,
                             comment="Renormalised density")
            return newdname
        else:
            return dname


    def deproject_velocities(self, dname, inclin=90.0):
        """Deproject Velocity map if it exists

        Parameters
        ----------
        dname: str
            Name of the datamap to deproject
        inclin: float [90]
            Inclination in degrees
        """

        if dname in self.dmaps:
            self.dmaps[dname].deproject_velocities(inclin=inclin)
        else:
            print("ERROR: no such data name in this Map")

class Profile(object):
    """A Profile is a set of DataProfiles associated via the same R profile.
    It is used to describe radial dprofiles e.g., rotation curves.

    Attributes
    ----------
    R: numpy array [None]
        Input location radii
    data: numpy array [None]
        Input values
    edata: numpy array [None]
        Uncertainties
    order: int [0]
        Order of the velocity moment. -1 for 'others' (e.g., X, Y)
    """
    def __init__(self, R=None, ref_size=None,
                 pname=None, ptype="", **kwargs):
        """
        Args:
            R (numpy array): radii
            pname (str): name of the profile [None]
            ptype (str): type of the profile

            **kwargs:
                data (array): input data
                edata (array): uncertainties
                dname (str): name of the data
                ptype (str): type of the dataprofile
                flag (str): flag for the dataprofile
                order (int): order
                comment (str): comment to be attached [""]
        """
        # Empty dictionary for the moments
        self.dprofiles = AttrDict()

        # See if a dataprofile is provided
        self.Runit = kwargs.pop("Runit", dict_units['R'])
        self.pixel_scale = kwargs.pop("pixel_scale", 1.)

        # Get the list of suffixes which will be used to attach datasets
        dict_dprofs, prof_kwargs = analyse_data_kwargs(**kwargs)

        # First getting the shape of the data
        if ref_size is not None:
            self.size = ref_size
        elif R is not None:
            self.size = R.size
        else:
            # Look for the first data which exists
            data = None
            for dname in dict_dprofs.keys():
                if "data" in dict_dprofs[dname]:
                    data = dict_dprofs[dname]['data']
                    break
            if check._check_ifarrays([data]):
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
        self.pname = pname
        self.ptype = ptype
        self.overwrite = kwargs.pop("overwrite", False)
        self._force_dtype = kwargs.pop("force_dtype", False)

        # Add each datamap one by one
        for dname, dprof_kwargs in dict_dprofs.items():
            if 'dname' not in dprof_kwargs:
                dprof_kwargs['dname'] = dname
            self.add_dataprofile(**dprof_kwargs)

        if Rfinestep > 0:
            self.interpolate(newstep=Rfinestep)

    @property
    def _get_pixel_scale(self):
        return (1. * u.pixel).to(self.Runit, equivalencies=self.eq_pscale).value

    def _convert_to_runit(self):
        """Convert XYunit into the default one
        a priori arcseconds.
        """
        self.R *= self.Runit_per_pixel
        # Update the unit
        self.Runit = dict_units['R']

    @property
    def Runit_per_pixel(self):
        return (1. * self.Runit).to(dict_units['R'],
                                 self.eq_pscale).value
    @property
    def eq_pscale(self):
        return u.pixel_scale(self.pixel_scale * self.Runit / u.pixel)

    def _init_R(self, R):
        """Initialise Rin

        Args:
            R: numpy array
        """
        # Define the grid in case Rin
        # If it is the case, using the reference profile
        if R is None:
            self.R = np.arange(self.size, dtype=default_float)
        else:
            self.R = R.ravel()
        self._convert_to_runit()

    @property
    def ndataprofs(self):
        return len(self.dprofiles)

    def _fullname(self, dname):
        return add_suffix(self.pname, dname, separator=default_data_separator)

    def _has_dataprofile(self, dname):
        return dname in self.dprofiles.keys()

    def _get_dataprofile(self, dname=None, order=None):
        """Get the dataprofile if it exists, and
        check the order

        Args:
            dname:
            order:

        Returns:

        """
        if dname is None:
            if order is None:
                # then just get the first profile
                dname = list(self.dprofiles.keys())[0]
            else:
                # Then get the first profile of right order
                for key in self.dprofiles.keys():
                    if self.dprofiles[key].order == order:
                        dname = key
                        break

        if self._has_dataprofile(dname):
            return self.dprofiles[dname]
        else:
            print("No such dataprofile {} in this Map".format(dname))
            return None

    def attach_dataprofile(self, dataprofile):
        """Attach a DataProfile to this Profile

        Args:
            dataprofile: DataProfile to attach
        """
        if self._check_dprofiles(dataprofile):
            self.dprofiles[dataprofile.dname] = dataprofile

    def add_dataprofile(self, **kwargs):
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
        data = kwargs.pop("data", None)
        if data is None:
            print("ERROR[add_dataprofile]: data is None - Aborting")
            return

        # Input dname to define the data. If none, define using the counter
        dname = kwargs.pop("dname", None)
        if dname is None or dname=="":
            dname = "{0}{1:02d}".format(self.pname, self.ndataprofs+1)
        if dname[0] == default_data_separator:
            dname = dname[1:]

        if self._has_dataprofile(dname) and not overwrite:
            print("WARNING[add_dataprofile]: data profile {} already exists "
                  "- Aborting".format(dname))
            print("WARNING[add_dataprofile]: use overwrite option to force.")
            return

        # Check if we wish to force the dtype / dunit
        force_dtype = kwargs.pop("force_dtype", self._force_dtype)
        if force_dtype:
            dtype = kwargs.get("dtype", None)
            dmap_info = get_dmap_info_from_dtype(dtype, dname)
            # Transfer
            for key, value in dmap_info.items():
                kwargs[key] = value

        self.attach_dataprofile(DataProfile(data=data, dname=dname, **kwargs))

    def __getattr__(self, dname):
        for suffix in default_data_names:
            if dname.startswith(suffix):
                for profname in self.dprofiles.keys():
                    if profname in dname:
                        basename = remove_suffix(dname, profname)
                        return getattr(self.dprofiles[profname], basename)
        raise AttributeError("'Profile' object has no attribute {}".format(dname))

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

    def interpolate(self, dname, step=1.0, suffix="fine", overwrite=False):
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
        if not self._has_dataprofile(dname):
            print("ERROR[interpolate]: no such dataprofile "
                  "with name {}".format(dname))
            return

        if hasattr(self.dprofiles[dname], add_suffix("R", suffix)):
            if overwrite:
                print("WARNING: overwriting existing interpolated profile")
            else:
                print("ERROR[interpolate]: interpolated profile exists. "
                      "Use 'overwrite' to update.")
                return

        Rfine, dfine, edfine = transform.interpolate_profile(self.R,
                                                             self.dprofiles[dname].data,
                                                             self.dprofiles[dname].edata,
                                                             step=step)
        setattr(self.dprofiles[dname], add_suffix("R", suffix), Rfine)
        setattr(self.dprofiles[dname], add_suffix("data", suffix), dfine)
        setattr(self.dprofiles[dname], add_suffix("edata", suffix), edfine)

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


def match_datamaps(map1, map2=None, dname1=None, dname2=None,
                   odname1=None, odname2=None, PAnodes=0.):
    """Aligning two datamaps

    Args:
        map1 (Map): input Map
        map2 (Map): second input Map. If None, use the first one.
        dmap1_name (str): name of input datamap 1
        dmap2_name (str): name of input datamap 2
        omap1_name (str): name of output datamap 1
        omap2_name (str): name of output datamap 2

    Returns:
        New Map with matched datamaps
    """
    if map2 is None:
        map2 = map1

    # Get the datamaps
    dmap1 = map1._get_datamap(dname1)
    dmap2 = map2._get_datamap(dname2)
    if dmap1 is None or dmap2 is None:
        return None

    # Determine the new grid
    XYextent = get_extent(map1.X_lon, map1.Y_lon)
    newstep = guess_stepxy(map1.X_lon, map1.Y_lon)
    Xn, Yn = np.meshgrid(cover_linspace(XYextent[0], XYextent[1], newstep),
                         cover_linspace(XYextent[2], XYextent[3], newstep))

    # Regrid
    new_data1 = transform.regrid_Z(map1.X_lon, map1.Y_lon, dmap1.data, Xn, Yn)
    new_edata1 = transform.regrid_Z(map1.X_lon, map1.Y_lon, dmap1.edata, Xn, Yn)
    new_data2 = transform.regrid_Z(map2.X_lon, map2.Y_lon, dmap2.data, Xn, Yn)
    new_edata2 = transform.regrid_Z(map2.X_lon, map2.Y_lon, dmap2.edata, Xn, Yn)

    # And re-attach to a regrided mass map
    omname1 = add_suffix(map1.mname, remap_suffix, separator="")
    if odname1 is None:
        odname1 = add_suffix(dmap1.dname, remap_suffix, separator="")
    if odname2 is None:
        odname2 = add_suffix(dmap2.dname, remap_suffix, separator="")
    mtype1 = add_prefix(map1.mtype, remap_suffix, separator="")
    dtype1 = dmap1.dtype.lower()
    dtype2 = dmap2.dtype.lower()

    # Creating the new Map
    print("INFO[match_datamaps]: Creating the first map {0} and "
          "attaching first datamap {1}".format(omname1, dname1))
    newMap = Map(mname=omname1, data=new_data1, edata=new_edata1, order=0,
                 mtype=mtype1, X=Xn, Y=Yn, dtype=dtype1, flag=dmap1.flag,
                 dname=odname1, alpha_north=-90.0-PAnodes)

    # Adding the second datamap
    print("INFO[match_datamaps]: attaching the datamap {0} to map {1}".format(
           dname2, omname1))
    newMap.add_datamap(data=new_data2, order=dmap2.order, edata=new_edata2,
                    dname=odname2, flag=dmap2.flag, dtype=dtype2,
                    dunit=dmap2.dunit)

    return newMap
