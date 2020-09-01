# -*- coding: utf-8 -*-
"""
This provides the basis for torque computation
"""
import numpy as np

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

from .disc import GalacticDisc
from .misc_io import add_suffix, AttrDict
from .transform import extract_radial_profile_fromXY
from .maps_grammar import remap_suffix, _is_flag_density
from .local_units import km_pc
from . import fit_functions as ff
from . import gravpot_functions as gpot

class TorqueMap(object):
    def __init__(self, massmap, mass_dname, comp_dname,
                 velprof, vel_dname, fac_pc=1.0, PAnodes=0.0,
                 factor_hz=12.0):
        self.massmap = massmap
        self.mass_dname = mass_dname
        self.comp_dname = comp_dname
        self.velprof = velprof
        self.vel_dname = vel_dname
        self.factor_hz = factor_hz

        self.dmass = self.massmap.dmaps[self.mass_dname]
        self.dcomp = self.massmap.dmaps[self.comp_dname]
        self.dvel = self.velprof.dprofiles[self.vel_dname]
        self.fac_pc = fac_pc

        # Check the coordinates
        self.Xdep = self.massmap.X_londep
        self.Ydep = self.massmap.Y_londep
        self.Rdep = np.sqrt(self.Xdep**2 + self.Ydep**2)
        self.Xdep_pc = self.Xdep * self.fac_pc
        self.Ydep_pc = self.Ydep * self.fac_pc
        self.Rdep_pc = self.Rdep * self.fac_pc
        self.pixel_scale = self.massmap.pixel_scale
        self.pc_per_pixel = self.pixel_scale * self.fac_pc
        self.PAnodes = PAnodes

    @property
    def XY_extent(self):
        return [np.min(self.Xdep), np.max(self.Xdep),
                np.min(self.Ydep), np.max(self.Ydep)]

    @property
    def XYpc_extent(self):
        return [np.min(self.Xdep_pc), np.max(self.Xdep_pc),
                np.min(self.Ydep_pc), np.max(self.Ydep_pc)]

    def get_mass_profile(self):
        """Compute the 1d mass profile
        """
        self.Rmass1d, self.mass1d = extract_radial_profile_fromXY(self.Xdep, self.Ydep,
                                                                  self.dmass.data,
                                                                  nbins=None, verbose=True,
                                                                  wedge_size=0.0, wedge_angle=0)

    def fit_mass_profile(self):
        """Fit the 1d mass profile
        """
        self.opt_mass, self.fsphe1d, self.fdisc1d, self.fmass1d = ff.fit_disc_sphe(self.Rmass1d, self.mass1d)
        self.bfit_mass1d = self.fmass1d(self.Rmass1d, self.opt_mass[0])
        self.bfit_mass1d_sphe = self.fsphe1d(self.Rmass1d, self.opt_mass[0])
        self.bfit_mass1d_disc = self.fdisc1d(self.Rmass1d, self.opt_mass[0])
        self.bfit_sphe1d = self.fsphe1d(self.massmap._R, self.opt_mass[0])
        self.bfit_sphe1d_dep = self.fsphe1d(self.Rdep, self.opt_mass[0])
        self.massmap.faceon = self.dmass.data - self.bfit_sphe1d_dep + self.bfit_sphe1d
        self.Rl_disc = self.opt_mass[0][3]

    def get_kernel(self, softening=0.0, function="sech2"):
        """Get the kernel array
        """
        hz_pc = self.Rl_disc * self.fac_pc / self.factor_hz
        self.kernel = gpot.get_gravpot_kernel(self.Rdep_pc, hz_pc,
                                              pc_per_pixel=self.pc_per_pixel,
                                              softening=softening,
                                              function=function)

    def get_gravpot(self):
        """Calculate the gravitational potential
        """
        self.gravpot = gpot.get_potential(self.massmap.faceon, self.kernel)

    def get_forces(self):
        """Calculate the forces from the potential
        """
        self.Fgrad, self.Fx, self.Fy, self.Frad, self.Ftan = \
            gpot.get_forces(self.Xdep_pc, self.Ydep_pc,
                            self.gravpot, self.PAnodes)

    def get_vrot_from_forces(self):
        """Calculate the velocities from forces
        """
        self.VcU = gpot.get_vrot_from_force(self.Rdep_pc, self.Frad)

    def get_torque_map(self):
        """Compute the torque map
        """
        self.torque_map = gpot.get_torque(self.Xdep_pc, self.Ydep_pc,
                                          self.Fx, self.Fy)

    def get_weighted_torque_map(self):
        """Compute the torque map
        """
        self.torque_w_map = gpot.get_weighted_torque(self.Xdep_pc, self.Ydep_pc,
                                          self.Fx, self.Fy, self.dcomp.data)

    def get_torque_profiles(self, n_rbins=300):
        """Get the profiles from the torque
        """
        self.r_mean, self.v_mean, self.torque_mean, self.torque_mean_w, \
            self.ang_mom_mean, self.dm, self.dm_sum = \
            gpot.get_torque_profiles(self.Xdep_pc, self.Ydep_pc, self.VcU,
                                     self.Fx, self.Fy, self.dcomp.data,
                                     n_rbins=n_rbins, pc_per_pixel=self.pc_per_pixel)
        # T in s
        self.Trot = 2. * np.pi * self.r_mean * km_pc / self.v_mean
        # dloL adimensional
        self.dloL = self.torque_mean_w * self.Trot / self.ang_mom_mean

    def get_torques(self, n_rbins=300):
        """Calculate the torques from existing forces

        Args:
            n_rbins:
        """
        self.get_torque_map()
        self.get_weighted_torque_map()
        self.get_torque_profiles(n_rbins=n_rbins)

    def run_torques(self, softening=0.0, func_kernel="sech2", n_rbins=200):
        """Running the torque calculation from start to end

        Args:
            softening:
            func_kernel:
            n_rbins:

        """
        # Step 1 - extract the radial profile and fit it with bulge and disc
        self.get_mass_profile()

        # Step 2 - Now doing the fit of the spheroid (and disc)
        self.fit_mass_profile()

        # Step 3 - calculate the kernel
        self.get_kernel(softening=softening, function=func_kernel)

        # Step 4 - calculate the potential
        self.get_gravpot()

        # Step 5 - Calculate the forces
        self.get_forces()

        # Step 6 - Get the rotation velocities
        self.get_vrot_from_forces()

        # Step 7 - Normalise the fields with M/L
        # For the moment = passed

        # Step 8 - Calculate the torques
        self.get_torques(n_rbins=n_rbins)


class GalacticTorque(GalacticDisc):
    """Class for functionalities associated with Torques
    """

    def __init__(self, vcfile_name=None, vcfile_type="ROTCUR",
                 Rfinestep=0, vprof_name="Velocity", **kwargs):
        """

        Args:
            vcfile_name:
            vcfile_type:
            **kwargs:
        """
        self.verbose = kwargs.pop("verbose", False)

        # Using GalacticDisc class attributes
        super().__init__(**kwargs)

        # Now the velocity file
        if vcfile_name is not None:
            print("INFO: Adding the provided Vc file")
            velname = self.add_vprofile(filename=vcfile_name, filetype=vcfile_type,
                              Rfinestep=Rfinestep, vprof_name=vprof_name)
        else:
            velname = kwargs.pop("velname", "vel_vel01")

        # And checking the maps
        compname = kwargs.pop("compname", "comp_comp01")
        massname = kwargs.pop("massname", "mass_mass01")
        self.init_torque_components(velname=velname, compname=compname,
                                    massname=massname)

        # Make sure we start with a clean set of torque maps
        self._reset_torquemaps()


    @property
    def velname(self):
        return self.profiles[self.vel_pname]._fullname(self.vel_dname)

    @property
    def massname(self):
        return self.maps[self.mass_mname]._fullname(self.mass_dname)

    @property
    def compname(self):
        return self.maps[self.comp_mname]._fullname(self.comp_dname)

    def init_torque_components(self, **kwargs):
        """Initialise the torque maps components

        Args:
            **kwargs: velname, compname, massname

        """
        self._decode_torque_names(**kwargs)
        self.check_torque_components()
        if self._check_all:
            self.match_comp_mass()

    def _decode_torque_names(self, velname=None, compname=None, massname=None):
        """

        Args:
            velname (str): composite name for the velocity profile
            compname (str): composite name for the component map
            massname (str): composite name for the mass map

        """
        # Decoding the names
        self.vel_pname, self.vel_dname = self._decode_prof_name(velname)
        self.comp_mname, self.comp_dname = self._decode_map_name(compname)
        self.mass_mname, self.mass_dname = self._decode_map_name(massname)

    def check_torque_components(self):
        """Find the components for the torque calculations using the composite
        names for the velocities, component (e.g., gas flux) and mass.

        """
        # Checking the maps and profiles
        # And making sure they are density maps
        if self._check_mass():
            thismap = self.massmap
            if not _is_flag_density(self.massdmap.flag):
                self.mass_dname = thismap.intmap_to_densitymap(self.mass_dname, self)

        if self._check_comp():
            thismap = self.compmap
            if not _is_flag_density(self.compdmap.flag):
                self.comp_dname = thismap.intmap_to_densitymap(self.comp_dname, self)

    @property
    def veldprof(self):
        return self.profiles[self.vel_pname].dprofiles[self.vel_dname]

    @property
    def velprof(self):
        return self.profiles[self.vel_pname]

    @property
    def compdmap(self):
        return self.maps[self.comp_mname].dmaps[self.comp_dname]

    @property
    def compmap(self):
        return self.maps[self.comp_mname]

    @property
    def massdmap(self):
        return self.maps[self.mass_mname].dmaps[self.mass_dname]

    @property
    def massmap(self):
        return self.maps[self.mass_mname]

    @property
    def _matched(self):
        return (self.mass_mname == self.comp_mname)

    @property
    def _check_all(self):
        return all((self._check_comp(), self._check_mass(), self._check_vel()))

    def _check_vel(self):
        return self._has_profile_data(self.vel_pname, self.vel_dname,
                                      order=1)

    def _check_comp(self):
        return self._has_map_data(self.comp_mname, self.comp_dname,
                                  order=0)

    def _check_mass(self):
        return self._has_map_data(self.mass_mname, self.mass_dname,
                                  order=0)

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

    def match_comp_mass(self, odname1="dmass", odname2="dcomp", **kwargs):
        """Aligning the gas onto the mass map
        """
        if not self._check_all:
            print("WARNING[match_comp_mass]: cannot proceed with match "
                  "as all maps are not yet set up. Please check")
            return

        if not self._matched:
            match_name = self.match_datamaps(self.mass_mname, self.comp_mname,
                                             self.mass_dname, self.comp_dname,
                                             odname1, odname2)
            print("INFO[match_comp_mass]: new map is {}".format(match_name))
            self.mass_mname = match_name
            self.comp_mname = match_name
            self.mass_dname = odname1
            self.comp_dname = odname2
            self.deproject_nodes(match_name)
        else:
            print("WARNING[match_comp_mass]: nothing to match as the data are "
                  "associated with the same map {}".format(self.mass_mname))

    def run_torques(self, torquemap_name=None, softening=0, func_kernel="sech2",
                    n_rbins=200, **kwargs):
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

        # Step 0 - finding the gas, mass and vel if not defined
        # and match the maps
        if not self._check_all:
            # Try to read the maps
            velname = kwargs.pop("velname", self.velname)
            compname = kwargs.pop("compname", self.compname)
            massname = kwargs.pop("massname", self.massname)
            self.init_torque_components(velname=velname, compname=compname,
                                        massname=massname)

        fac_pc = self.pc_per_xyunit(self.massmap.XYunit)

        # Defining the structure in which the torques will be calculated
        # Note that PAnodes is now -90 as we put the nodes along the X axis - horizontal
        # During the matching
        newT = TorqueMap(self.massmap, self.mass_dname, self.comp_dname,
                         self.velprof, self.vel_dname, fac_pc=fac_pc,
                         PAnodes=-90.0)

        # Running the torque calculation
        newT.run_torques(softening=softening, n_rbins=n_rbins,
                         func_kernel=func_kernel)

        # Now allocating it to the torquemaps
        self.tmaps[torquemap_name] = newT
