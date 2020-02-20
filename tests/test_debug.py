# from pydisc.torques import GalaxyTorques
# n6951folder = "/soft/python/pytorque/examples/data_6951/"
# mass6951 = pyfits.getdata(n6951folder+"r6951nicmos_f160w.fits")
# gas6951 = pyfits.getdata(n6951folder+"co21-un-2sigma-m0.fits")
# gas6951 = gas6951.reshape(gas6951.shape[0]*gas6951.shape[1], gas6951.shape[2])
# vc6951 = "rot-co21un-01.tex"
# t51 = GalaxyTorques(vcfile_name=n6951folder+vc6951, vcfile_type="ROTCUR",
#                           mass=mass6951, gas=gas6951, Xcenmass=178.0, Ycenmass=198.0,
#                           Xcengas=148.0, Ycengas=123.0, inclination=41.5, distance=35.0,
#                           PA=138.7, stepXgas=0.1, stepYgas=0.1, stepXmass=0.025, stepYmass=0.025)
# import pydisc
from pydisc.density_wave import DensityWave

# Importing useful modules
from astropy.io import fits as pyfits
from os.path import join as joinpath
import numpy as np

# Getting the data
ddata = "/home/science/PHANGS/MUSE/MUSEDAP/"
n1512 = "NGC1512_MAPS.fits"
# Open the Maps files
maps = pyfits.open(joinpath(ddata, n1512))
# Extract the mass, flux, and velocity maps
mass = maps['STELLAR_MASS_DENSITY'].data
flux = maps['FLUX'].data
vel = maps['V_STARS'].data

mydisc = DensityWave(data_flux=flux, edata_flux=np.zeros_like(flux),
                     data_mass=mass, data_vel=vel, edata_vel=np.zeros_like(vel),
                     name="MUSE", Xcen=462.5, Ycen=464.4, PA_nodes=90)