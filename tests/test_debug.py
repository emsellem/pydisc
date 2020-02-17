from pydisc.torques import GalaxyTorques

n6951folder = "/soft/python/pytorque/examples/data_6951/"
mass6951 = pyfits.getdata(n6951folder+"r6951nicmos_f160w.fits")
gas6951 = pyfits.getdata(n6951folder+"co21-un-2sigma-m0.fits")
gas6951 = gas6951.reshape(gas6951.shape[0]*gas6951.shape[1], gas6951.shape[2])
vc6951 = "rot-co21un-01.tex"
t51 = GalaxyTorques(vcfile_name=n6951folder+vc6951, vcfile_type="ROTCUR",
                          mass=mass6951, gas=gas6951, Xcenmass=178.0, Ycenmass=198.0,
                          Xcengas=148.0, Ycengas=123.0, inclination=41.5, distance=35.0,
                          PA=138.7, stepXgas=0.1, stepYgas=0.1, stepXmass=0.025, stepYmass=0.025)