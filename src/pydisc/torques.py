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
from .misc_io import add_suffix, AttrDict
from .misc_io import extract_radial_profile_fromXY
from . import fit_functions as ff
from . import gravpot_functions as gpot

dict_torques_datanames = {"gasd": "DFlux", "gas": "Flux", "mass": "Mass", "massd": "DMass"}

class TorqueMap(object):
    def __init__(self, mass_name, gas_name, vel_name):
        self.gas_name = gas_name
        self.mass_name = mass_name
        self.vel_name = vel_name

class GalaxyTorques(GalacticDisc):
    """Class for functionalities associated with Torques
    """

    def __init__(self, vcfile_name=None, vcfile_type="ROTCUR",
                 **kwargs):
        """

        Args:
            gasd:
            mass:
            vcfile_name:
            vcfile_type:
            **kwargs:
        """
        self.verbose = kwargs.pop("verbose", False)

        # Using GalacticDisc class attributes
        super().__init__(**kwargs)

        # Look for the reference X, Y
        self._get_XY(**kwargs)
        Rfinestep = kwargs.pop("Rfinestep", 0)

        if self.Xrefname is None:
            print("WARNING: no X,Y coordinates provided")

        print("INFO: Adding the provided components")
        self.add_components(**kwargs)

        if vcfile_name is not None:
            print("INFO: Adding the provided Vc file")
            self.add_vprofile(filename=vcfile_name, filetype=vcfile_type,
                              Rfinestep=Rfinestep)

        self._reset_torquemaps()

    def _get_XY(self, **kwargs):
        """Extract first X,Y attributes and save names and
        values

        Args
            **kwargs: set of arguments

        """
        # First look for the reference input coordinates
        self.Xref = None
        self.Yref = None
        self.Xrefname = None
        self.Yrefname = None
        for kwarg in kwargs:
            if kwarg.startswith("X") and not kwarg.startswith("Xcen"):
                self.Xrefname = kwarg
                self.Yrefname = kwarg.replace("X", "Y")
                self.Xref = kwargs.get(self.Xrefname, None)
                self.Yref = kwargs.get(self.Yrefname, None)
                break

    def add_components(self, overwrite=False, **kwargs):
        """Decipher the new components and add them
        """
        # We go for the various datasets
        # Loop over the potential names
        for key in dict_torques_datanames.keys():
            # Loop over the given arguments
            list_kwarg = [kwarg for kwarg in kwargs if kwarg.startswith(key)]
            for item in list_kwarg:
                # First removing that argument = data
                data = kwargs.pop(item, None)
                # if X argument not found, use refX
                nameX = add_suffix("X", item)
                if nameX not in kwargs:
                    Xin, Yin = self.Xref, self.Yref
                else:
                    # Otherwise just extract the X and Y
                    Xin = kwargs.pop(nameX, None)
                    Yin = kwargs.pop(add_suffix("Y", item), None)
                # Now trace all remaining items which end like mykwarg
                list_pop = [kwarg for kwarg in kwargs if kwarg.endswith(item)]
                # And loop over these to create the local kwargs
                mykwargs = {}
                for fitem in list_pop:
                    name_item = fitem.replace(item, "")
                    mykwargs[name_item] = kwargs.pop(fitem, None)
                print("INFO: Adding the {} component".format(item))
                self.add_component(data, name=item,
                                   type=key, Xin=Xin, Yin=Yin,
                                   overwrite=overwrite, **mykwargs)

    def add_component(self, data, name=None, Xin=None, Yin=None,
                      type=None, flag=None, **kwargs):
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
        if type not in dict_torques_datanames.keys():
            print("ERROR[add_component]: type of data not recognised - Aborting")
            return

        kwargs[dict_torques_datanames[type]] = data
        if name is None:
            name = "{}_data".format(type)

        self.add_map(name=name, order=0, type=type,
                     Xin=Xin, Yin=Yin,
                     data_type=type, data_flag=flag,
                     data_name=name, **kwargs)

    def _reset_torquemaps(self):
        """Initalise the torquemap Profiles by setting an empty
        'torquemaps' dictionary
        """
        # set up the torquemaps
        self.tmaps = AttrDict()

    @property
    def ntorquemaps(self):
        """Number of existing torquemap profiles
        """
        if hasattr(self, 'tmaps'):
            return len(self.tmaps)
        else:
            return -1

    def run_torques(self, gas_name=None, mass_name=None, vel_name=None, torquemap_name=None):
        """Running the torques recipes

        Args:
            gas_name:
            mass_name:
            vel_name:
            torquemap_name:

        Returns:

        """

        if torquemap_name is None:
            torquemap_name = "Torq{0:02d}".format(self.ntorquemaps+1)

        # Defining the structure in which the torques will be calculated
        newT = TorqueMap(mass_name, gas_name, vel_name)

        # Step 0 - finding the gas, mass and vel if not defined
        if gas_name is None:
            gas_map = self._get_map(type="gas")
        if mass_name is None:
            mass_map = self._get_map(type="mass")
        if vel_name is None:
            vel_prof = self._get_profile(type="vel")

        # Step 1 - Deproject the disc mass map
        self.deproject_nodes(mass_map.name)

        # Step 2 - extract the radial profile and fit it with bulge and disc
        Xdep, Ydep = mass_map.X_londep, mass_map.Y_londep
        Rdep = np.sqrt(Xdep**2 + Ydep**2)
        newT.Rmass1d, newT.mass1d = extract_radial_profile_fromXY(Xdep, Ydep, mass_map.M.data,
                                                                     nbins=None, verbose=True,
                                                                     wedge_size=0.0, wedge_angle=0)

        # Step 3 - Now doing the fit of the spheroid (and disc)
        opt_par, bestfit = ff.fit_spheroid(newT.Rmass1d, newT.mass1d)
        newT.fit_mass1d = bestfit(newT.Rmass1d, opt_par[0])
        newT.opt_fit = opt_par[0]

        # Calculating the spheroid on the projected map
        newT.fit_spheroid = bestfit(mass_map._Rin, np.append(opt_par[0][:2], [0,0]))
        newT.fit_spheroid_dep = bestfit(Rdep, np.append(opt_par[0][:2], [0,0]))

        # Subtracting that spheroid while adding the deprojected one
        newT.mass_faceon = mass_map.M.data - newT.fit_spheroid_dep + newT.fit_spheroid

        # Step 4 - calculate the kernel
        Rdep_pc = Rdep * self.pc_per_arcsec
        newT.kernel = gpot.get_gravpot_kernel(Rdep_pc, softening=0, function="sech2")

        # Step 5 - calculate the potential
        newT.gravpot = gpot.get_potential(newT.mass_faceon, newT.kernel)

        # Step 6 - Calculate the forces
        newT.Fgrad, newT.Fx, newT.Fy, newT.Frad, newT.Ftan = get_forces(Xdep_pc, Ydep_pc, newT.gravpot)

        # Step 7 - Normalise the fields with M/L

        # Step 8 - Calculate the torques

        # Now allocating it to the torquemaps
        self.tmaps[torquemap_name] = newT
