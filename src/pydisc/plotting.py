# -*- coding: utf-8 -*-
"""
A set of useful functions to derive the gravitational potential
and to help with deprojection
"""

__version__ = '1.1.2 (04-12, 2019)'

# Changes --
#   04/12/19- EE - v1.1.2: transferred from pytorque
#   25/06/19- EE - v1.1.0: Python 3
#   13/04/07- EE - v1.0.1: Addition of stop_program()

"""
Import of the required modules
"""
# Numpy
import numpy as np

# Other important packages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import cmocean

from .misc_functions import clipping_to_rectangle
from .misc_io import  add_suffix

def visualise_data(Xin, Yin, Zin, newextent=None,
                   fill_value=np.nan, method='linear',
                   verbose=False, newstep=None,
                   **kwargs):
    """Visualise a data set via 3 input arrays, Xin, Yin and Zin.
    The shapes of these arrays should be the same.

    Input
    =====
    Xin, Yin, Zin: 3 real arrays
    newextent : requested extent [xmin, xmax, ymin, ymax]
        for the visualisation. Default is None
    fill_value : Default is numpy.nan
    method : 'linear' as Default for the interpolation, when needed.
    """
    extent, newX, newY, newZ = resample_data(Xin, Yin, Zin,
                                             newextent=newextent,
                                             newstep=newstep,
                                             fill_value=fill_value,
                                             method=method,
                                             verbose=verbose)
    plt.clf()
    plt.imshow(newZ, extent=extent, **kwargs)
    return extent, newX, newY, newZ

def show_tw(disc, slicing_name=None, map_name=None,
            coef=4, vminV=-150, vmaxV=150, live=True,
            flag=None, **kwargs):
    """Makes an interactive plot of the results from
    applying the Tremaine Weinberg method.

    Args:
        disc: GalacticDisc
        slicing_name: str [None]
        coef: int [4]
        vminV: float [-150]
        vmaxV: float [150]
            Minimum and maximum of velocities for the
            plot.
        live: bool [True]

    """

    # get the slicing
    slicing = disc._get_slicing(slicing_name)
    thismap = disc._get_map(map_name)

    # Set up colourbar limits for the image
    Iname = kwargs.pop("Iname", add_suffix("I", flag))
    Vname = kwargs.pop("Vname", add_suffix("V", flag))
    Flux = getattr(thismap.dmaps, Iname).data
    Vel = getattr(thismap.dmaps, Vname).data
    sel_flux = (Flux != 0)
    vminF = np.nanpercentile(np.log10(Flux[sel_flux]), 0.5)
    vmaxF = np.nanpercentile(np.log10(Flux[sel_flux]), 99.75)

    # Starting the figure
    fig = plt.figure(figsize=(6, 6), tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    # START of the x2 axis ---------------------
    ax1 = fig.add_subplot(gs[0,0])
    ax1.imshow(np.log10(Flux),
               cmap=cmocean.cm.gray_r,
               origin='lower', interpolation='none',
               vmin=vminF, vmax=vmaxF,
               extent=thismap.XYin_extent)

    ncolour_plot = len(slicing.ycentres[::coef])
    # Setting the colour cycle
    colour_plot = cmocean.cm.thermal(np.linspace(0, 1, ncolour_plot))
    ax1.set_prop_cycle(color=colour_plot)

    # Loop over the slices with 1 out of coef slits
    nslits = len(slicing.ycentres)

    # Get all the slits
    pa = thismap._get_angle_from_PA(disc.PA_nodes)
    alpha_rad = np.deg2rad(pa + 90.0)

    # lines_inter will be the x, y coordinates for the slit
    lines_inter = np.zeros((nslits, 2, 2))
    for i in range(nslits):
        lines_inter[i] = clipping_to_rectangle(slicing.ycentres[i], pa, extent=thismap.XYin_extent)

    ycmin, ycmax = np.min(slicing.yedges), np.max(slicing.yedges)

    # --------------------------------------------------------------
    # Two functions to pick up the data from the line slit plot
    # and the scatter plot
    def line_picker(scat, mouseevent):
        """
        find the slit within a certain distance from the mouseclick in
        data coords and then update the ax3 plot to show which line it is
        """
        if mouseevent.xdata is None:
            return False, dict()

        yslit = mouseevent.ydata * np.cos(alpha_rad) \
                - mouseevent.xdata * np.sin(alpha_rad)
        # Finds the closest slit
        ind = np.abs(slicing.ycentres - yslit).argmin()
        if (yslit >= ycmin) and (yslit <= ycmax):
            # Finds the scatter point
            pickx, picky = scat.get_offsets().data[ind]
            props = dict(ind=ind, pickx=pickx, picky=picky)
            return True, props
        else:
            return False, dict()

    def scat_picker(scat, mouseevent):
        """
        find the points within a certain distance from the mouseclick in
        data coords and then update the ax1 plot to show which line it is
        """
        if mouseevent.xdata is None:
            return False, dict()

        xdata, ydata = scat.get_offsets().data.T
        maxd = 0.05
        d = np.sqrt(
            ((xdata - mouseevent.xdata)/range_dfx)**2
            + ((ydata - mouseevent.ydata)/range_dfV)**2)

        ind = np.argmin(d)
        if (d[ind] <= maxd) :
            pickx = xdata[ind]
            picky = ydata[ind]
            props = dict(ind=ind, pickx=pickx, picky=picky)
            return True, props
        else:
            return False, dict()
    # --------------------------------------------------------------

    # Plot with colour cycle
    line1 = ax1.plot([lines_inter[::coef, 0, 0], lines_inter[::coef, 1, 0]],
                     [lines_inter[::coef, 0, 1], lines_inter[::coef, 1, 1]],
                     linewidth=1, picker=line_picker)

    # Create a reference slit which is hidden
    refline = ax1.plot([lines_inter[0, 0, 0], lines_inter[0, 1, 0]],
                       [lines_inter[0, 0, 1], lines_inter[0, 1, 1]],
                       'r-', linewidth=3, zorder=4)
    refline[0].set_visible(False)

    # START of the x2 axis ---------------------
    ax2 = fig.add_subplot(gs[0,1])
    ax2.imshow(Vel,
               cmap=cmocean.cm.balance,
               origin='lower', interpolation='none',
               vmin=vminV, vmax=vmaxV,
               extent=thismap.XYin_extent)

    # Also plot on the axis line
    line_y0 = clipping_to_rectangle(0, pa, extent=thismap.XYin_extent)
    ax2.plot([line_y0[0,0], line_y0[1,0]], [line_y0[0,1], line_y0[1,1]], 'k--', linewidth=2)

    # START of the 3rd plot with the scatter points
    ax3 = fig.add_subplot(gs[1, :])
    nerr = len(slicing.dfx_tw)
    range_dfx, range_dfV = np.max(slicing.dfx_tw), np.max(slicing.dfV_tw)
    colour_scat = cmocean.cm.thermal(np.linspace(0, 1, nerr))

    # All functions for the events themselves ===========================
    def onpick(event):
        print('Coordinate of closest point = ', event.pickx, event.picky)

    def update_refline(ind):
        refline[0].set_ydata([lines_inter[ind,0,1], lines_inter[ind,1,1]])
        refline[0].set_xdata([lines_inter[ind, 0, 0], lines_inter[ind, 1, 0]])

    def update_point(x, y):
        scatref.set_offsets([[x, y]])

    def hover(event):
        if event.inaxes in [ax1, ax2]:
            vis_s = scatref.get_visible()
            vis_l = refline[0].get_visible()
            cont, prop = line_picker(scat, event)
            if cont:
                update_point(prop['pickx'], prop['picky'])
                update_refline(prop['ind'])
                scatref.set_visible(True)
                refline[0].set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis_s or vis_l:
                    scatref.set_visible(False)
                    refline[0].set_visible(False)
                    fig.canvas.draw_idle()

        if event.inaxes == ax3:
            vis = refline[0].get_visible()
            cont, prop = scat_picker(scat, event)
            if cont:
                update_refline(prop['ind'])
                update_point(prop['pickx'], prop['picky'])
                scatref.set_visible(True)
                refline[0].set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    scatref.set_visible(False)
                    refline[0].set_visible(False)
                    fig.canvas.draw_idle()
    # End of event functions =====================================

    # first errorbars
    for i in range(nerr):
        ax3.errorbar(slicing.dfx_tw[i], slicing.dfV_tw[i],
                     xerr=slicing.dfx_tw_err[i],
                     yerr=slicing.dfV_tw_err[i],
                     marker="none", fmt="none",
                     linestyle='none', c=colour_scat[i],
                     zorder=0)

    # The points themselves
    scat = ax3.scatter(slicing.dfx_tw, slicing.dfV_tw,
                       c=colour_scat, zorder=3, picker=scat_picker)
    # Plot the reference point and not visible at start
    scatref = ax3.scatter(slicing.dfx_tw[1],
                          slicing.dfV_tw[1], c='red', zorder=4)
    scatref.set_visible(False)

    # Some label
    ax3.set_xlabel(r'<$x$> $\left(^{\prime \prime}\right)$')
    ax3.set_ylabel(r'<$v$> $\left(\mathrm{km\,s}^{-1}\right)$')

    # Finally connect the event
    if live:
        fig.canvas.mpl_connect('pick_event', onpick)
        fig.canvas.mpl_connect("motion_notify_event", hover)

