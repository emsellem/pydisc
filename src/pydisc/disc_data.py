# -*- coding: utf-8 -*-
"""
Module for the data classes: datamap or profile
"""

# External modules
import numpy as np
from numpy import deg2rad

# Float
from .misc_io import add_suffix, default_float, add_err_prefix

# Check
from . import check, transform

class datamap(object):
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

class dataset(object):
    """A dataset is a set of values associated with a location grid (X, Y)
    It is used to describe a set of e.g., velocity fields, flux maps, etc
    A grid is associated natively to these datamaps as well as an orientation
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
        Type of the dataset
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
        self.attach_data(data, **kwargs)
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
        # If it is the case, using the reference dataset
        if Xin is None or Yin is None:
            ref_ind = np.indices(self.shape, dtype=default_float)
            if Xin is None: Xin = ref_ind[1]
            if Yin is None: Yin = ref_ind[0]

        if not check._check_consistency_sizes([Xin, Yin]):
            print("ERROR: errors on sizes of Xin and Yin")
            return False
        else:
            self.shape = Xin.shape
            # Making sure the shapes agree
            Yin = Yin.reshape(self.shape)
            self._nameX = add_suffix(nameX, self.flag)
            self._nameY = add_suffix(nameY, self.flag)
            self.Xin = Xin
            self.Yin = Yin
            return True

    def attach_data(self, data, order=0, edata=None, data_name="", data_attr_name=""):
        """Attach a new datamap to the present dataset. Will check if
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

        datamap_to_attach = datamap(data, edata, order, data_name)
        self.datamaps[name_data_key] = datamap_to_attach
        setattr(self, name_data_key, self.datamaps[name_data_key])

        setattr(self, data_attr_name, self.datamaps[name_data_key].data)
        setattr(self, add_err_prefix(data_attr_name), self.datamaps[name_data_key].edata)

    def check_XYin(self):
        """Check if Xin and Yin are ok and consistent
        """


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

            # Reshaping input dataset into the same shape
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

    def align_xy_NorthEast(self) :
        """Get North to the top and East on the left
        """
        self.X_NE, self.Y_NE = self.rotate(matrix=self._mat_NE)

    def align_xy_lineofnodes(self, galaxy) :
        """Set the Line of Nodes (defined by its Position Angle, angle from the North
        going counter-clockwise) as the positive X axis
        """
        self.X_lon, self.Y_lon = self.rotate(matrix=galaxy._mat_lon * self._mat_NE)

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

class profile(object):
    def __init__(self):
        pass
