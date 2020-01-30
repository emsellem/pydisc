# -*- coding: utf-8 -*-
"""
This provides the basis for torque computation
"""
import numpy as np
import os

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

from .disc import GalacticDisc
from .misc_io import read_vc_file, add_suffix
from .disc_data import DataSet1D

dict_torques_datanames = {"gasd": "DFlux", "gas": "Flux", "mass": "Mass", "massd": "DMass"}

class GalaxyTorques(GalacticDisc):
    """Class for functionalities associated with Torques
    """

    def __init__(self, vc_filename=None, vcfile_type="ROTCUR",
                 **kwargs):
        """

        Args:
            gasd:
            mass:
            vc_filename:
            vcfile_type:
            **kwargs:
        """
        self.verbose = kwargs.pop("verbose", False)

        # Using GalacticDisc class attributes
        super().__init__(**kwargs)

        # Look for the reference X, Y
        Xrefname, Yrefname, Xref, Yref = self._get_XY(**kwargs)
        Rfinestep = kwargs.pop("Rfinestep", 0)

        if Xrefname is None:
            print("WARNING: no X,Y coordinates provided")
        else:
            print("INFO: Adding the provided components")
            self.add_components(**kwargs)

        if vc_filename is not None:
            print("INFO: Adding the provided Vc file")
            self.read_vc_file(vc_filename, vcfile_type, Rfinestep)

    def add_components(self, **kwargs):
        """Decipher the new components and add them
        """
        # We go for the various datasets
        # Loop over the potential names
        for key in dict_torques_datanames.keys():
            # Loop over the given arguments
            for kwarg in kwargs:
                # if the argument start right then
                if kwarg.startswith(key):
                    # Isolate the suffix and put it in the datasetname
                    name = kwarg.replace(key, "")
                    # if X argument not found, use refX
                    nameX = add_suffix("X", key)
                    if nameX not in kwargs:
                        Xin, Yin = Xref, Yref
                    else:
                        # Otherwise just extract the X and Y
                        Xin = kwargs.get(add_suffix("X", key), None)
                        Yin = kwargs.get(add_suffix("Y", key), None)
                    data = kwargs.get(key, None)
                    flag = kwargs.get(add_suffix(key, "flag"), None)
                    print("INFO: Adding the {} component".format(name))
                    self.add_component(data, name=name, flag=flag,
                                       type=key, Xin=Xin, Yin=Yin)

    def read_vc_file(self, filename, vcfile_type="ROTCUR", Rfinestep=0):
        status, radius, Vc, eVc = read_vc_file(filename,
                                               Vcfile_type=vcfile_type)
        self.Vprof = DataSet1D(name=filename, Rin=radius,
                                 data=Vc, prof_name="Vc", Rfinestep=Rfinestep)

    def _get_XY(self, **kwargs):
        """Extract first X,Y attributes and send back the names and
        values

        Args
            **kwargs: set of arguments

        Returns
            nameX, nameY, X, Y
        """
        # First look for the reference input coordinates
        counter = 0
        for kwarg in kwargs:
            if kwarg.startswith("X"):
                if counter == 0:
                    Xrefname = kwarg
                    Yrefname = kwarg.replace("X", "Y")
                    Xref = kwargs.get(Xrefname, None)
                    Yref = kwargs.get(Yrefname, None)
                counter += 1

        if counter == 0:
            return [None]*4

        return Xrefname, Yrefname, Xref, Yref

    def add_component(self, data, name=None, Xin=None, Yin=None,
                      type=None, flag=None, overwrite=False):
        """Adding a dataset (gas or mass) to the GalaxyTorque

        Args:
            data:
            name:
            type:
            Xin:
            Yin:
            flag:
            overwrite:

        Returns:

        """
        mykwarg = {"Xin": Xin, "Yin":Yin}
        if type not in dict_torques_datanames.keys():
            print("ERROR[add_component]: type of data not recognised - Aborting")
            return

        mykwarg[dict_torques_datanames[type]] = data
        if name is None:
            name = "{}_data".format(type)
        mykwarg["map_name"] = name

        if not self._has_dataset(name):
            self.datasets[name].attach_datamap(data, order=0,
                                               map_type=type,
                                               map_flag=flag,
                                                **mykwargs)
