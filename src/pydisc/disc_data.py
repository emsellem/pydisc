# -*- coding: utf-8 -*-
"""
Module for the data classes: DataMap or profile
"""

# External modules
import numpy as np
from numpy import deg2rad

# Float
from .misc_io import add_suffix, default_float, add_err_prefix
from .misc_io import AttrDict, remove_suffix
from .misc_io import default_suffix_separator, default_prefix_separator
from . import check, transform

default_data_names = ["data", "edata"]

class DataMap(object):
    """Data class representing a specific map

    Attributes
    ----------
    data
    edata
    order
    name
    """
    def __init__(self, data, edata=None, order=0, name=None, flag=None, type=""):
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
            type: str [""]
        """
        self.data = data
        self.edata = edata
        self.order = order
        self.name = name
        self.flag = flag
        self.type = type

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

class DataSet(object):
    """A DataSet is a set of values associated with a location grid (X, Y)
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
    Xin, Yin: numpy arrays [None, None]
        Input location grid
    data: numpy array [None]
        Input values
    edata: numpy array [None]
        Uncertainties
    order: int [0]
        Order of the velocity moment. -1 for 'others' (e.g., X, Y)
    """
    def __init__(self, Xin=None, Yin=None, ref_shape=None,
                 name=None, nameX="X", nameY="Y", **kwargs):
        """
        Args:
            Xin: numpy array [None]
                Input X axis location array
            Yin: numpy array [None]
                Input Y axis location array
            Xcen, Ycen: float, float
                Centre for the X and Y axes. Default is the centre of the image.
            ref_shape: tuple [None]
                Reference shape for the arrays. If provided, will be used to shape the
                input arrays
            name: str [None]
                Name of the dataset
            nameX: str ["X"]
                Name for the X coordinate.
            nameY: str ["Y"]
                Name for the X coordinate.

            **kwargs:
                data: array
                edata: array [None]
                    Uncertainty map
                map_name: str
                    Name of the map
                map_type: str
                    Type of the map
                map_flag: str
                    Flag for the map
                order: int
                    Order for the datamap
                comment: str [""]
                    Comment attached to the dataset
        """
        # Empty dictionary for the moments
        self.datamaps = AttrDict()

        # See if a datamap is provided
        data = kwargs.pop("data", None)

        # First getting the shape of the data
        if ref_shape is not None:
            self.shape = ref_shape
        elif Xin is not None:
            self.shape = Xin.shape
        elif data is not None:
            self.shape = data.shape

        self.Xcen = kwargs.pop("Xcen", self.shape[0] / 2.)
        self.Ycen = kwargs.pop("Ycen", self.shape[1] / 2.)
        if not self._init_XY(Xin, Yin, nameX, nameY):
            print("ERROR: Xin and Yin are not compatible - Aborting.")
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

        # Comment
        self.comment = kwargs.pop("comment", "")
        self.name = name

        # Attaching a given datamap
        if data is not None:
            self.attach_datamap(data, **kwargs)
            self.check_datamap()

    def __getattr__(self, name):
        for suffix in default_data_names:
            if name.startswith(suffix):
                for mapname in self.datamaps.keys():
                    if mapname in name:
                        basename = remove_suffix(name, mapname)
                        return getattr(self.datamaps[mapname], basename)

    def __dir__(self, list_names=default_data_names):
        return  [add_suffix(attr, map) for item in list_names
                for map in self.datamaps.keys() for attr in self.datamaps[map].__dir__()
                if attr.startswith(item)]

    def _init_XY(self, Xin, Yin, nameX="X", nameY="Y"):
        """Initialise Xin and Yin

        Args:
            Xin: numpy array
            Yin: numpy array
                Input X, Y grid.
            nameX:
            nameY:
        """
        # Define the grid in case Xin, Yin not yet defined
        # If it is the case, using the reference DataSet
        if Xin is None or Yin is None:
            ref_ind = np.indices(self.shape, dtype=default_float)
            if Xin is None: Xin = ref_ind[1]
            if Yin is None: Yin = ref_ind[0]

        if not check._check_consistency_sizes([Xin, Yin]):
            print("ERROR: errors on sizes of Xin and Yin")
            return False
        else:
            # Making sure the shapes agree
            Yin = Yin.reshape(self.shape)
            self._nameX = nameX
            self._nameY = nameY
            self.Xin = Xin - self.Xcen
            self.Yin = Yin - self.Ycen
            self.XYin_extent = [np.min(self.Xin), np.max(self.Xin),
                                np.min(self.Yin), np.max(self.Yin)]
            return True

    def _has_datamap(self, name):
        return name in self.datamaps.keys()

    def attach_datamap(self, data, order=0, edata=None,
                       map_name=None, map_flag=None, map_type=None,
                       overwrite=False):
        """Attach a new DataMap to the present DataSet. Will check if
        grid is compatible.

        Args:
            data: 2d array
            order: int
            edata: 2d array
            map_name: str
            map_type: str
            map_flag: str

        """
        if data is None:
            return

        # Input name to define the data. If none, define using the counter
        if map_name is None:
            map_name = "map{0:02d}".format(len(self.datamaps)+1)

        if self._has_datamap(map_name) and not overwrite:
            print("WARNING: data map {} already exists - Aborting".format(
                    map_name))
            print("WARNING: use overwrite option to force.")
            return

        self.datamaps[map_name] = DataMap(data, edata, order,
                                          map_name, map_flag, map_type)

    def check_datamap(self, name_datamap="all"):
        """Check consistency of data
        """
        if name_datamap == "all":
            list_datamaps = list(self.datamaps.keys())
        else:
            list_datamaps = [name_datamap]

        # Putting everything in 1D
        ref_array = self.Xin.ravel()

        # Main loop on the names of the datamaps
        for name in list_datamaps:
            data = self.datamaps[name].data
            edata = self.datamaps[name].edata
            arrays_to_check = [data.ravel()]
            if edata is not None:
                arrays_to_check.append(edata.ravel())
            if not check._check_ifarrays(arrays_to_check):
                print("ERROR: input datamaps {} not all arrays".format(
                    name))
                self.datamaps.pop(name)
                continue

            arrays_to_check.insert(0, ref_array)
            if not check._check_consistency_sizes(arrays_to_check):
                print("ERROR: input maps {} have not the same size "
                      "than input grid (Xin, Yin)".format(name))
                self.datamaps.pop(name)
                continue

            # Reshaping input DataSet into the same shape
            # And adding it to the dictionary
            self.datamaps[name].data = data.reshape(self.shape)
            self.datamaps[name].edata = edata.reshape(self.shape)

    def align_axes(self, galaxy):
        """Align all axes using Xin and Yin as input
        """
        self.align_xy_lineofnodes(galaxy)
        self.align_xy_bar(galaxy)
        self.align_xy_deproj_bar(galaxy)

    @property
    def Xin(self):
        return self.__Xin

    @Xin.setter
    def Xin(self, Xin):
        setattr(self, self._nameX, Xin)
        self.__Xin = Xin

    @property
    def Yin(self):
        return self.__Yin

    @Yin.setter
    def Yin(self, Yin):
        setattr(self, self._nameY, Yin)
        self.__Yin = Yin

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
        X,Y set of arrays using self.Xin and self.Yin

        Parameters
        ----------
        **kwargs: set of arguments, see transform.rotate
            Includes Xin, Yin, matrix

        Returns:
        The rotated arrays Xrot, Yrot
        """
        X = kwargs.pop("X", self.Xin)
        Y = kwargs.pop("Y", self.Yin)
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

        if map_name in self.datamaps:
            self.datamaps[map_name].deproject_velocities(inclin=inclin)
        else:
            print("ERROR: no such data name in this DataSet")


class DataSet1D(object):
    """A DataSet1D is a set of Profiles associated via the same R profile.
    It is used to describe radial profiles e.g., rotation curves.

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
    def __init__(self, Rin=None, ref_size=None,
                 name=None, **kwargs):
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
                prof_name: str
                    Name of the profile
                prof_type: str
                    Type of the profile
                prof_flag: str
                    Flag for the profile
                order: int
                    Order for the profile
                comment: str [""]
                    Comment attached to the dataset1d
        """
        # Empty dictionary for the moments
        self.profiles = AttrDict()

        # See if a datamap is provided
        data = kwargs.pop("data", None)

        # First getting the shape of the data
        if ref_size is not None:
            self.size = ref_size
        elif Rin is not None:
            self.size = Rin.size
        elif data is not None:
            self.size = data.size

        self._init_R(Rin)

        # Filling value
        self._fill_value = kwargs.pop("fill_value", 'nan')
        # Method
        self._method = kwargs.pop("method", "linear")

        # Comment
        self.comment = kwargs.pop("comment", "")
        self.name = name

        # New step in R when provided
        Rfinestep = kwargs.pop("Rfinestep", 0)

        # Attaching a given datamap
        if data is not None:
            self.attach_profile(data, **kwargs)
            self.check_profiles()

        if Rfinestep > 0:
            self.interpolate(newstep=Rfinestep)

    def _init_R(self, Rin):
        """Initialise Rin

        Args:
            Rin: numpy array
            nameR:
        """
        # Define the grid in case Rin
        # If it is the case, using the reference profile
        if Rin is None:
            self.Rin = np.arange(self.size, dtype=default_float)
        else:
            self.Rin = Rin.ravel()

    def _has_profile(self, name):
        return name in self.profiles.keys()

    def attach_profile(self, data, order=0, edata=None,
                       prof_name=None, prof_flag=None, prof_type=None,
                       overwrite=False):
        """Attach a new Profile to the present Set.

        Args:
            data: 1d array
            order: int
            edata: 1d array
            prof_name: str
            prof_type: str
            prof_flag: str

        """
        if data is None:
            return

        # Input name to define the data. If none, define using the counter
        if prof_name is None:
            prof_name = "prof{0:02d}".format(len(self.profiles)+1)

        if self._has_profile(prof_name) and not overwrite:
            print("WARNING: data profile {} already exists - Aborting".format(
                profile_name))
            print("WARNING: use overwrite option to force.")
            return

        self.profiles[prof_name] = DataProfile(data, edata, order=order,
                                               name=prof_name, flag=prof_flag,
                                               type=prof_type)
        setattr(self, prof_name, self.profiles[prof_name])

    def __getattr__(self, name):
        for suffix in default_data_names:
            if name.startswith(suffix):
                for profname in self.profiles.keys():
                    if profname in name:
                        basename = remove_suffix(name, profname)
                        return getattr(self.profiles[profname], basename)

    def __dir__(self, list_names=default_data_names):
        return  [add_suffix(attr, prof) for item in list_names
                for prof in self.profiles.keys() for attr in self.profiles[prof].__dir__()
                if attr.startswith(item)]

    def check_profiles(self, name_profile="all"):
        """Check consistency of data
        """
        if name_profile == "all":
            list_profiles = list(self.profiles.keys())
        else:
            list_profiles = [name_profile]

        # Putting everything in 1D
        ref_array = self.Rin

        # Main loop on the names of the datamaps
        for name in list_profiles:
            data = self.profiles[name].data
            edata = self.profiles[name].edata
            arrays_to_check = [data.ravel()]
            if edata is not None:
                arrays_to_check.append(edata.ravel())
            if not check._check_ifarrays(arrays_to_check):
                print("ERROR: input profiles {} not all arrays".format(
                    name))
                self.profiles.pop(name)
                continue

            arrays_to_check.insert(0, ref_array)
            if not check._check_consistency_sizes(arrays_to_check):
                print("ERROR: input profiles {} have not the same size "
                      "than input grid (Rin)".format(name))
                self.profiles.pop(name)
                continue

    def interpolate(self, step=1.0, suffix="fine", overwrite=False):
        """Provide interpolated profile

        Args:
            stepR: float [1.0]
            suffix: str [""]
            overwrite: bool [False]

        Returns:

        """
        if step <= 0:
            print("ERROR[interpolate]: new step is <= 0 - Aborting")
        # Getting the data
        if self.data is None:
            print("ERROR[interpolate]: input data is None - Aborting")
            return

        if hasattr(self, add_suffix("R", suffix)):
            if overwrite:
                print("WARNING: overwriting existing interpolated profile")
            else:
                print("ERROR[interpolate]: interpolated profile exists. "
                      "Use 'overwrite' to update.")
                return

        Rfine, dfine, edfine = transform.interpolate_profile(self.Rin,
                                                             self.data,
                                                             self.edata,
                                                             step=step)
        setattr(self, add_suffix("R", suffix), Rfine)
        setattr(self, add_suffix("data", suffix), dfine)
        setattr(self, add_suffix("edata", suffix), edfine)

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
                type: ""
        """
        # Using DataMap class attributes
        super().__init__(data=data, edata=edata, **kwargs)

        # Now 1D in case the input is 2D
        if data is not None:
            self.data = self.data.ravel()
        if edata is not None:
            self.edata = self.edata.ravel()

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

