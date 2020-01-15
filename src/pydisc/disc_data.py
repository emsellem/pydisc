# -*- coding: utf-8 -*-
"""
Module for the data classes: DataMap or profile
"""

# External modules
import numpy as np
from numpy import deg2rad

# Float
from .misc_io import add_suffix, default_float, add_err_prefix
from . import check, transform

class DataMap(object):
    """Data class representing a specific map

    Attributes
    ----------
    data
    edata
    order
    name
    flag
    """
    def __init__(self, data, edata=None, order=0, name=""):
        """
        Args:
            data: numpy array
                Input datas values.
            edata: numpy array [None]
                Uncertainties for the data.
            order: int [0]
                order of the velocity moment. Can be -1 for 'others' (grid)
            name: str ['']
                Name of the attribute
            flag: str ['']
                Name to flag the map (e.g., 'CO', 'Ha')
        """
        self.data = data
        self.edata = edata
        self.order = order
        self.name = name

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
    data_type: str ['']
        Type of the DataSet
    flag: str [""]
        Flag for the data (e.g., 'CO', 'Ha')
    """
    def __init__(self, Xin=None, Yin=None, ref_shape=None,
                 nameX="X", nameY="Y", **kwargs):
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
            nameX: str ["X"]
                Name for the X coordinate. The flag will be used as suffix.
            nameY: str ["Y"]
                Name for the X coordinate. The flag will be used as suffix.

            **kwargs:
                flag: str ['']
                    Flag for the data
                data
                order
                edata
                data_name
                data_attr_name
                edata_attr_name
        """
        # Empty dictionary for the moments
        self.datamaps = {}
        self.flag = kwargs.pop("flag", None)
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
        self.dataset_name = kwargs.pop("dataset_name", None)

        # Attaching the given dataset
        self.attach_datamap(data, **kwargs)
        self.check_data()

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
            self._nameX = add_suffix(nameX, self.flag)
            self._nameY = add_suffix(nameY, self.flag)
            self.Xin = Xin - self.Xcen
            self.Yin = Yin - self.Ycen
            self.XYin_extent = [np.min(self.Xin), np.max(self.Xin),
                                np.min(self.Yin), np.max(self.Yin)]
            return True

    def attach_datamap(self, data, order=0, edata=None,
                       data_name="", data_attr_name="", **kwargs):
        """Attach a new DataMap to the present DataSet. Will check if
        grid is compatible.

        Args:
            data:
            order:
            edata:
            data_name:

        """
        if data is None:
            return

        # Input name to define the data. If none, define using the counter
        if data_name is None:
            data_name = "data{0:02d}".format(len(self.datamaps)+1)
        if self.flag is None:
            name_data_key = data_name
        else:
            name_data_key = add_suffix(data_name, self.flag)

        self.datamaps[name_data_key] = DataMap(data, edata, order, data_name)
        setattr(self, name_data_key, self.datamaps[name_data_key])

        setattr(self, data_attr_name, self.datamaps[name_data_key].data)
        setattr(self, add_err_prefix(data_attr_name), self.datamaps[name_data_key].edata)

    def check_data(self, name_datamap="all"):
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

    def align_axes(self):
        """Align all axes using Xin and Yin as input
        """
        self.align_xy_lineofnodes()
        self.align_xy_bar()
        self.align_xy_deproj_bar()

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

    def deproject_velocities(self, name_data_key):
        """Deproject Velocity values by dividing by the sin(inclination)
        """

        if name_data_key in self.datamaps:
            datamap = self.datamaps[name_data_key]
            if datamap.order != 1:
                print("ERROR: data are not order 1 velocities -- Aborting")
                return
            Vdep = transform.deproject_velocities(datamaps.data,
                                                  self.inclin)
            eVdep = transform.deproject_velocities(datamap.edata,
                                                  self.inclin)
            name_data = "{}_dep".format(datamaps.name)
            self.datamaps[name_data_key_dep] = DataMap(Vdep, eVdep, 1, name_data)
        else:
            print("ERROR: no such data name in this DataSet")

class Profile(object):
    def __init__(self):
        pass

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

