test=2
if test == 1:
    from pydisc.torques import GalacticTorque
    from astropy.io import fits as pyfits
    n6951folder = "/soft/python/pytorque/examples/data_6951/"
    mass6951 = pyfits.getdata(n6951folder+"r6951nicmos_f160w.fits")
    gas6951 = pyfits.getdata(n6951folder+"co21-un-2sigma-m0.fits")
    gas6951 = gas6951.reshape(gas6951.shape[0]*gas6951.shape[1], gas6951.shape[2])
    vc6951 = "rot-co21un-01.tex"

# #tmp = GalacticDisc(datamass=mass6951, datagas=gas6951,
# #                   dtypemass="Mass", dtypegas="Weight",
# #                   Xcenmass=178.0, Ycenmass=198.0,
# #                   Xcengas=148.0, Ycengas=123.0,
# #                   inclination=41.5, distance=35.0,
# #                   PA_nodes=138.7,
# #                   pixel_scalegas=0.1, pixel_scalemass=0.025)
    t51 = GalacticTorque(vcfile_name=n6951folder+vc6951, vcfile_type="ROTCUR",
                         datamass=mass6951, datagas=gas6951, Xcenmass=178.0, Ycenmass=198.0,
                         Xcengas=148.0, Ycengas=123.0, inclination=41.5, distance=35.0,
                         PA_nodes=138.7, pixel_scalegas=0.1, pixel_scalemass=0.025)
    t51.run_torques()

if test == 2:
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

    mydisc = DensityWave(data_flux=flux, data_mass=mass, data_vel=vel,
                         name="MUSE", Xcen=462.5, Ycen=464.4, PAnodes=90)
    mydisc.tremaine_weinberg(slit_width=5.0, map_name="MUSE")
    o = mydisc.fit_slope_tw()
#    mydisc.plot_tw()