# -*- coding: utf-8 -*-
"""
This provides the basis for torque computation
"""
import numpy as np
from astropy import units as u

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

from .disc import GalacticDisc
from .misc_io import add_suffix, AttrDict, add_err_prefix, add_prefix
from .misc_io import extract_radial_profile_fromXY
from .misc_io import get_extent, guess_stepxy, cover_linspace
from . import fit_functions as ff
from . import gravpot_functions as gpot
from .transform import regrid_Z
from .disc_data import dict_units

dict_torques_datanames = {"gas": "Weight", "gasd": "WeightD",
                          "weightd": "WeightD", "weight": "Weight",
                          "mass": "Mass", "massd": "MassD"}
dict_torques_griddatanames = {"gridgasd": "FluxD", "gridmassd": "MassD"}
density_prefix = "d"
def _is_density(name):
    return name.lower().endswith(density_prefix)

def _add_density_suffix(name):
    return add_suffix(name, density_prefix, link="")

class TorqueMap(object):
    def __init__(self, mass, vel, fac_pc=1.0):
        self.mass = mass
        self.vel = vel
        self.fac_pc = fac_pc

        # Check the coordinates
        self.Xdep = self.mass.X_londep
        self.Ydep = self.mass.Y_londep
        self.Rdep = np.sqrt(self.Xdep**2 + self.Ydep**2)
        self.Xdep_pc = self.Xdep * self.fac_pc
        self.Ydep_pc = self.Ydep * self.fac_pc
        self.Rdep_pc = self.Rdep * self.fac_pc
        self.pixel_scale = self.mass.pixel_scale
        self.pc_per_pixel = self.pixel_scale * self.fac_pc

    def get_mass_profile(self):
        """Compute the 1d mass profile
        """
        self.Rmass1d, self.mass1d = extract_radial_profile_fromXY(self.Xdep, self.Ydep,
                                                                  self.mass.data_mass,
                                                                  nbins=None, verbose=True,
                                                                  wedge_size=0.0, wedge_angle=0)

    def fit_mass_profile(self):
        """Fit the 1d mass profile
        """
        self.opt_mass, self.fsphe1d, self.fdisc1d, self.fmass1d = ff.fit_disc_sphe(self.Rmass1d, self.mass1d)
        self.bfit_mass1d = self.fmass1d(self.Rmass1d, self.opt_mass[0])
        self.bfit_mass1d_sphe = self.fsphe1d(self.Rmass1d, self.opt_mass[0])
        self.bfit_mass1d_disc = self.fdisc1d(self.Rmass1d, self.opt_mass[0])
        self.bfit_sphe1d = self.fsphe1d(self.mass._R, self.opt_mass[0])
        self.bfit_sphe1d_dep = self.fsphe1d(self.Rdep, self.opt_mass[0])
        self.mass.faceon = self.mass.data_mass - self.bfit_sphe1d_dep + self.bfit_sphe1d
        self.Rl_disc = self.opt_mass[0][3]

    def get_kernel(self, softening=0.0, function="sech2"):
        """Get the kernel array
        """
        hz_pc = self.Rl_disc * self.fac_pc / 24.
        self.kernel = gpot.get_gravpot_kernel(self.Rdep_pc, hz_pc,
                                              pc_per_pixel=self.pc_per_pixel,
                                              softening=softening,
                                              function=function)

    def get_gravpot(self):
        """Calculate the gravitational potential
        """
        self.gravpot = gpot.get_potential(self.mass.faceon, self.kernel)

    def get_forces(self):
        """Calculate the forces from the potential
        """
        self.Fgrad, self.Fx, self.Fy, self.Frad, self.Ftan = gpot.get_forces(self.Xdep * self.fac_pc,
                                                                             self.Ydep * self.fac_pc,
                                                                             self.gravpot)
    def get_vrot_from_forces(self):
        """Calculate the velocities from forces
        """
        self.VcU = gpot.get_vrot_from_force(self.Rdep_pc, self.Frad)

    def get_torques(self, n_rbins=50):
        """Calculate the torques from existing forces

        Args:
            n_rbins:

        Returns:

        """
        self.r_mean, self.v_mean, self.torque_mean, self.torque_mean_w, \
                self.ang_mom_mean, self.dl, self.dm, self.dm_sum = \
                gpot.get_torque(self.Xdep_pc, self.Ydep_pc, self.VcU, self.Fx, self.Fy,
                                self.mass.data_WeightD, n_rbins=n_rbins)

    def run_torques(self, softening=0.0, func_kernel="sech2", n_rbins=50):
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

        # Step 3 - resample to a standard grid with squared pixel
        # And go to mass density before resampling
        # Come back to Mass before going to the gravitational potential

        # Step 4 - calculate the kernel
        self.get_kernel(softening=softening, function=func_kernel)

        # Step 5 - calculate the potential
        self.get_gravpot()

        # Step 6 - Calculate the forces and velocities
        self.get_forces()
        self.get_vrot_from_forces()

        # Step 7 - Normalise the fields with M/L

        # Step 8 - Calculate the torques
        self.get_torques(n_rbins=n_rbins)

class GalacticTorque(GalacticDisc):
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
        super().__init__(read_maps=False, force_dtypes=False, **kwargs)

        # Look for the reference X, Y
        self._get_XY(**kwargs)
        Rfinestep = kwargs.pop("Rfinestep", 0)

        # Just a warning that the refname is not provided
        # Hence no X, Y => will be initialised from indices
        if self.Xrefname is None:
            print("WARNING: no X,Y coordinates provided")

        # Adding the provided components
        self.add_components(**kwargs)

        # Now the velocity file
        if vcfile_name is not None:
            print("INFO: Adding the provided Vc file")
            self.add_vprofile(filename=vcfile_name, filetype=vcfile_type,
                              Rfinestep=Rfinestep)

        # Make sure we start with a clean set of torque maps
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

    def add_components(self, **kwargs):
        """Decipher the new components and add them
        The only thing is to convert the pre-defined map keys into
        the right types
        """
        # Loop over the potential names for mass and gas maps
        list_map_kwargs  = self._analyse_kwargs_tobuild_maps(**kwargs)
        for map_kwargs in list_map_kwargs:
            # Overwrite the mtype when map name found
            map_name = map_kwargs['name']
            # Check if name is in the dictionary
            mtype = None
            for key in dict_torques_datanames.keys():
                if map_name.startswith(key):
                    mtype = key
                    break
            if mtype is not None:
                dtype = dict_torques_datanames[map_name]
                # Forcing the mtype
                map_kwargs['mtype'] = mtype
                # Change to density map if not already the case
                if not _is_density(dtype):
                    try:
                        XYunit = map_kwargs['XYunit']
                    except KeyError:
                        XYunit = dict_units['XY']
                    scalepc2 = (self.pc_per_xyunit(XYunit)) ** 2
                    map_kwargs['data'] /= scalepc2
                    if 'edata' in map_kwargs.keys():
                        if map_kwargs['edata'] is not None:
                            map_kwargs['edata'] /= scalepc2

                    map_kwargs['dtype'] = _add_density_suffix(dtype)
                self.add_maps(**map_kwargs)
            else:
                print("ERROR: map name not in dictionary {}"
                      "- Not adding this Map".format(map_name))

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

    def match_gas_and_mass(self, gas_name=None, mass_name=None):
        """Aligning the gas onto the mass map
        """
        gas_map = self._get_map(gas_name, mtype="gas")
        mass_map = self._get_map(mass_name, mtype="mass")

#        # Deproject the two maps
#        self.deproject_nodes(mass_map.name)
#        self.deproject_nodes(gas_map.name)

        # Get the datamap from gas
        mass_datamap = mass_map._get_datamap(order=0)

        # Get the datamap from gas
        gas_datamap = gas_map._get_datamap(order=0)

        # Determine the new grid
        XYextent = get_extent(mass_map.X_lon, mass_map.Y_lon)
        newstep = guess_stepxy(mass_map.X_lon, mass_map.Y_lon)
        Xn, Yn = np.meshgrid(cover_linspace(XYextent[0], XYextent[1], newstep),
                             cover_linspace(XYextent[2], XYextent[3], newstep))

        # Regrid
        newmass_data = regrid_Z(mass_map.X_lon, mass_map.Y_lon, mass_datamap.data,
                               Xn, Yn)
        newmass_edata = regrid_Z(mass_map.X_lon, mass_map.Y_lon, mass_datamap.edata,
                                Xn, Yn)
        newgas_data = regrid_Z(gas_map.X_lon, gas_map.Y_lon, gas_datamap.data,
                               Xn, Yn)
        newgas_edata = regrid_Z(gas_map.X_lon, gas_map.Y_lon, gas_datamap.edata,
                               Xn, Yn)

        # And re-attach to a regrided mass map
        name_mass = add_suffix(mass_map.name, "grid", link="")
        name_gas = add_suffix(gas_map.name, "grid", link="")
        type_mass = add_prefix(mass_map.mtype, "grid")
        type_gas = add_prefix(gas_map.mtype, "grid")
        dtype_gas = dict_torques_datanames[gas_datamap.dtype.lower()]
        dtype_mass = dict_torques_datanames[mass_datamap.dtype.lower()]

        mykwargs = {}
        mykwargs["data"] = newmass_data
        mykwargs["edata"] = newmass_edata
        # Adding the gridded mass map
        print("INFO[torques/match]: attaching the mass map")
        self.add_maps(name=name_mass, order=0, mtype=type_mass,
                     X=Xn, Y=Yn, dtype=dtype_mass, flag=mass_map.flag,
                     dname=mass_map.name, **mykwargs)

        # Deprojecting this one
        self.deproject_nodes(name_mass)

        dmap_info = self._extract_dmap_info_from_dtype(dtype_gas)
        # Adding the gas data
        print("INFO[torques/match]: attaching the data for the gas datamap")
        self.maps[name_mass].add_data(newgas_data, order=dmap_info['order'],
                          edata=newgas_edata, dname=dmap_info['dname'],
                          flag=dmap_info['flag'],
                          dtype=dmap_info['dtype'],
                          dunit=dmap_info['dunit'])

        return self.maps[name_mass]

    def run_torques(self, gas_name=None, mass_name=None, vel_name=None,
                    torquemap_name=None, softening=0, func_kernel="sech2",
                    n_rbins=50):
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
        # and match the gas and mass maps
        mass_map = self.match_gas_and_mass(gas_name, mass_name)
        vel_prof = self._get_profile(vel_name, ptype="vel")

        fac_pc = self.pc_per_xyunit(mass_map.XYunit)

        # Defining the structure in which the torques will be calculated
        newT = TorqueMap(mass_map, vel_prof, fac_pc=fac_pc)

        # Running the torque calculation
        newT.run_torques(softening=softening, n_rbins=n_rbins,
                         func_kernel=func_kernel)

        # Now allocating it to the torquemaps
        self.tmaps[torquemap_name] = newT
