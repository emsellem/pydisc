#!/usr/bin/env python
# coding: utf-8

# # pydisc Notebook - 1.0

# ## 1 - Introduction

# *pydisc* is a python package meant to ease the use the manipulation of maps and profiles and the computation of basic quantities pertaining to galactic Discs. In this Notebook, I will show you how to use the main functionalities of *pydisc*.

# ## 2 - Structures in *pydisc*

# ### DataMaps, DataProfiles, Maps and Profiles

# In the language of *pydisc*:
# - **DataMaps** are data on a grid (hence 2D), e.g., values like velocities, flux, etc. The grid on which it is defined should be regular.
# - **DataProfiles** are data on a radial profile (hence 1D).
# - **Maps** are then a **set of** *DataMaps* associated with a set of coordinates X, Y.
# - **Profiles** are then a **set of** *DataProfiles* associated with a set of radial coordinates R.
# 
# DataMaps have orientations defined as 'NE_direct' indicating if the North-to-East axis is direct (counter-clockwise) or indirect (clockwise). It also has an 'alpha_North' angle which provides the angle between the North and the top (positive y-axis). DataMaps also have a pixelscale which provides the conversion between arcseconds and pixels in case the X,Y grids are not defined. If this is the case, X, and Y will be computed using just the indices from the grid.
# 
# DataMaps and DataProfiles have 'units' as defined by astropy units. These should be compatible with e.g., arcseconds, so these are observational. 
# 
# DataMap arguments:
# - dunit: astropy unit
# - order: velocity moment order. Hence velocities are order=1, flux or mass is 0, dispersion is 2, anything else would be -1 and there is a category for "dummy" maps with order=-10.
# - dname: name of the datamap
# - flag: a flag which is meant to add info (string)
# - data and edata: numpy arrays. If edata is not provided, it will be defined as None.
# 
# DataProfiles have similar arguments, but with punit (profile unit) and pname.
# 
# Maps arguments:
# - name: name of the map
# - X and Y: the 2 main arrays. If not provided, indices will be used.
# - Xcen and Ycen: centre for the 0,0
# - XYunit: unit (astropy) for the X and Y axis
# - NE_direct, alpha_North, etc.
# 
# Note that a Map can have many datamaps: hence a set of X,Y can have many data associated to it (sharing the same coordinates), each one having a different dname, order, flag etc.

# ### Galaxy

# A 'Galaxy' is an object which has some characteristics like: a distance, a Position Angle for the line of Nodes, an inclination (in degrees) and the Position Angle for a bar if present.

# ### GalacticDisc

# A 'GalacticDisc' is a structure associating a set of Maps and Profiles and a given Galaxy. 
# 
# This is the main structure which we will be using for the calculation of various quantities. 
# 
# There are a number of associated classes, namely:
# - 'DensityWave': associated with methods for density waves like the Tremaine Weinberg method
# - 'GalacticTorque': associated with methods for deriving torques
# 
# all inheriting from the GalacticDisc class, thus sharing a number of functionalities, but also have their own specific ones (which require a set of maps).
#     
# The 'grammar' for maps and datamaps is simple (a priori):
# - if you have an attribute like "data" you can input this in the argument list as:
# "data<map_name>_<datamap_name>". Hence if the map is name "MUSE" and the datamap named "vstar" you should have an argument for the data as "dataMUSE_vstar" and the associated "edataMUSE_vstar" if you have uncertainties for this map etc. Same applies for all argument of the maps and data, for example (using the same example): orderMUSE_vstar, XMUSE, YMUSE, XcenMUSE, YcenMUSE, flagMUSE_vstar...
# - In this way you can have several datamaps attached to a single map and have e.g.,: XMUSE, YMUSE, dataMUSE_vstar, dataMUSE_gas, dataMUSE_...
# 

# # 3- Examples

# ## 3.1 - Tremaine Weinberg

# Here is an example of how to get a Tremaine-Weinberg calculation made on a set of maps using the *DensityWave* class.

# In[2]:


# Importing the package and the DensityWave class
import pydisc
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


# In[3]:


# mname is for mapname. 
mydisc = DensityWave(data_flux=flux, edata_flux=np.zeros_like(flux),
                     data_mass=mass, data_vel=vel, edata_vel=np.zeros_like(vel),
                     mname="MUSE", Xcen=462.5, Ycen=464.4, PAnodes=90)


# In[4]:


# We can now look at the structure itself. 'mydisc' has a one map, which is named 'MUSE'. 
# This map is in a dictionary and is a Map class, as shown when printing it.
mydisc.maps


# In[5]:


# We can also find out about the other variables:
mydisc.maps['MUSE'].X


# In[6]:


# This Map actually has datamaps as shown here, each one having data.
# You can see that this Map has actually three datamaps, one with flux, one with mass, the last one with vel.
mydisc.maps['MUSE'].dmaps


# In[7]:


# We can call the data like this (note that the array shows the nan from the outer part of the map)
mydisc.maps['MUSE'].dmaps['flux'].data


# In[8]:


# or like this using the combined "data" with the name of the data map.
mydisc.maps['MUSE'].dmaps.flux.data


# In[9]:


# to make it simpler, the maps and dmaps are merged into one attribute automatically
mydisc.MUSE_flux


# In[10]:


mydisc.MUSE_flux.data


# In[11]:


# Now let's do the Tremaine Weinberg step. Defining slits of 5 arcsec.
# The programme will align the axes using the PA of the line of nodes as provided.
# The warning is just about nan and 0's being used in the division.
mydisc.tremaine_weinberg(slit_width=5.0, map_name="MUSE")


# In[12]:


# And you can now look at the result
print("Slicings: ", mydisc.slicings)
# Looking at the slicings
print("MUSE Slicing", mydisc.slicings['MUSE'])
# and its content
print("Yedges = ", mydisc.slicings['MUSE'].yedges)
print("Nslits?: ", mydisc.slicings['MUSE'].nslits)
print("Omega sinus(inclin) of TW method", mydisc.slicings['MUSE'].Omsini_tw)


# ## 3.2 Torques

# Now let's consider the other class inheriting from the GalacticDisc class, namely: GalacticTorque, which itself uses TorqueMap(s).

# In[13]:


from pydisc.torques import GalacticTorque

n6951folder = "/soft/python/pytorque/examples/data_6951/"
mass6951 = pyfits.getdata(n6951folder+"r6951nicmos_f160w.fits") 
gas6951 = pyfits.getdata(n6951folder+"co21-un-2sigma-m0.fits")
gas6951 = gas6951.reshape(gas6951.shape[0]*gas6951.shape[1], gas6951.shape[2])
vc6951 = "rot-co21un-01.tex"


# In[14]:


t51 = GalacticTorque(vcfile_name=n6951folder+vc6951, vcfile_type="ROTCUR",
                    datamass=mass6951, datacomp=gas6951, Xcenmass=178.0, Ycenmass=198.0,
                    Xcencomp=148.0, Ycencomp=123.0, inclination=41.5, distance=35.0,
                    PA_nodes=138.7, pixel_scalecomp=0.1, pixel_scalemass=0.025)


# You can then see what is in t51 structure:

# In[15]:


t51.maps


# In[16]:


t51.maps['mass'].dmaps


# In[17]:


from matplotlib import pyplot as plt
plt.imshow(t51.maps['mass'].dmaps['mass01'].data)


# In[18]:


plt.imshow(t51.maps['comp'].dmaps['comp01'].data)


# In[19]:


# Now running the torques
t51.run_torques()


# In[20]:


t51.tmaps


# In[21]:


plt.imshow(t51.tmaps['Torq01'].Fx)


# In[24]:


plt.imshow(t51.tmaps['Torq01'].torque_map)

