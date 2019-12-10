# -*- coding: utf-8 -*-
"""
This is the main class of the package - disc - gathering
data, attributes and functions defining a disc model.

This module include computation for torques, Tremaine-Weinberg
method, in-plane velocities (Maciejewski et al. method).
"""

import numpy as np
from numpy import deg2rad, rad2deg, cos, sin, arctan, tan, pi

from . import transform, misc_functions

class Galaxy(object):
    """
    Attributes
    ----------
    distance
    pc_per_arcsec
    inclin
    PA_nodes
    PA_bar
    """
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs:
                distance
                inclin
                PA_nodes
                PA_bar
        """

        self.distance = kwargs.pop('distance', 10.)
        self.pc_per_arcsec = misc_functions.get_pc_per_arcsec(self.distance)

        # Inclination in degrees
        self.inclin = kwargs.pop("inclination", 60.)
        # PA of the line of nodes
        self.PA_nodes = kwargs.pop("PA_nodes", 0.)
        # PA of the bar
        self.PA_bar = kwargs.pop("PA_bar", 0.)

    @property
    def inclin(self) :
        return self.__inclin

    @inclin.setter
    def inclin(self, inclin) :
        self.__inclin = inclin
        self.__inclin_rad = deg2rad(inclin)
        self._mat_inc = transform.set_stretchmatrix(coefY=1. / cos(self.__inclin_rad))

    @property
    def PA_nodes(self) :
        return self.__PA_nodes

    @PA_nodes.setter
    def PA_nodes(self, PA_nodes) :
        self.__PA_nodes = PA_nodes
        self.__PA_nodes_rad = deg2rad(PA_nodes)
        self._mat_lon = transform.set_rotmatrix(self.__PA_nodes_rad + pi / 2.)

    @property
    def PA_bar(self) :
        return self.__PA_bar

    @PA_bar.setter
    def PA_bar(self, PA_bar) :
        self.__PA_bar = PA_bar
        self.__PA_bar_rad = deg2rad(PA_bar)
        self.PA_barlon = PA_bar - self.PA_nodes
        self._PA_barlon_rad = deg2rad(self.PA_barlon)
        self._PA_barlon_dep_rad = arctan(tan(self._PA_barlon_rad) / cos(self.__inclin_rad))
        self._PA_barlon_dep = rad2deg(self._PA_barlon_dep_rad)
        self._mat_bar = transform.set_rotmatrix(self.__PA_bar_rad + pi / 2.)
        self._mat_bardep = transform.set_rotmatrix(self._PA_barlon_dep_rad)
