# -*- coding: utf-8 -*-
"""
This module provides the grammar used for maps, profiles etc
"""

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

# External modules
from collections import OrderedDict

# Local modules
from .misc_io import add_suffix, remove_suffix, add_prefix
from . import local_units as lu

dic_moments = OrderedDict([
    # order   dtype   INT/DEN/NONE  comment unit
    (-10, [("Dummy", "NA", "Dummy category", lu.pixel)]),
    (-1, [("Other", "NA", "Any category", lu.pixel)]),
    (0, [("Flux", "INT", "flux [integrated]", lu.Lsun),
         ("Mass", "INT", "mass [integrated]", lu.Msun),
         ("FluxD", "DENS", "flux density [per unit area]", lu.Lsunpc2),
         ("MassD", "DENS", "mass density [per unit area]", lu.Msunpc2),
         ("WeightD", "DENS", "To be used as density weights [per unit area]", lu.Lsunpc2),
         ("Weight", "INT", "To be used as weights [integrated]", lu.Lsun),
        ("CompD", "DENS", "To be used as density component [per unit area]", lu.Lsunpc2),
        ("Comp", "INT", "To be used as component [integrated]", lu.Lsun)]),
    (1, [("Vel", "NA", "velocity", lu.kms)]),
    (2, [("Disp", "NA", "dispersion", lu.kms),
         ("Mu2", "NA", "non-centred 2nd order moment", lu.kms2)])
])

dict_invert_moments = {}
for order in dic_moments.keys():
    for tup in dic_moments[order]:
        dict_invert_moments[tup[0].lower()] = (order,) + tup[1:]

# List of attributes to use for Maps
list_Map_attr = ["Xcen", "Ycen", "X", "Y", "mname", "mtype", "XYunit",
                 "pixel_scale", "NE_direct", "alpha_North", "fill_value",
                 "method", "overwrite"]

# List of attributes for Profiles
list_Profile_attr = ["R", "pname", "ptype", "Runit"]

# List of attributes to use for DataMaps
list_Data_attr = ["order", "dtype", "dname", "data", "edata", "dunit", "flag"]

density_suffix = 'd'
density_prefix = 'd'
density_flag = 'dens'
remap_suffix = 'grid'
default_mapprof_separator = ''
default_data_separator = '_'

def _is_dtype_density(name):
    return name.lower().endswith(density_prefix)

def _is_flag_density(name):
    return name.lower().startswith(density_flag)

def _add_density_suffix(name):
    return add_suffix(name, density_suffix, separator="")

def _add_density_prefix(name):
    return add_prefix(name, density_prefix, separator="")

def get_all_moment_types():
    """Get all types for all moments
    """
    return [get_moment_type(order).lower() for order in dic_moments.keys()]

def get_all_moment_tuples():
    """Get all tuples from dic_moments
    """
    return [tup for order in dic_moments.keys() for tup in dic_moments[order]]

def get_moment_type(order):
    """Returns all potential variable types for this order

    Args
    ----
    order: int
    """
    return [defset[0].lower() for defset in dic_moments[order]]

def get_dmap_info_from_dtype(dtype, dname=""):
    """ Force dmap info from dtype

    Args:
        self:
        dtype:

    Returns:
        dmap_info with the right keywords
    """
    # Initialise dmap_info
    dmap_info = {}
    if dtype is None:
        dtype = dname

    dict_dtype = extract_suff_from_keywords([dtype], dict_invert_moments.keys())
    if len(dict_dtype) > 0:
        # For all tuples in pre-defined types
        thistype = list(dict_dtype.values())[0][0]
        dmap_info['comment'] = dict_invert_moments[thistype][2]
        dmap_info['flag'] = dict_invert_moments[thistype][1]
        dmap_info['dtype'] = thistype
        dmap_info['order'] = dict_invert_moments[thistype][0]
        dmap_info['dunit'] = dict_invert_moments[thistype][3]
    return dmap_info

def analyse_data_kwargs(map_name="", **kwargs):
    list_Data_attr_up = [add_suffix(dattr, map_name,
                                    separator=default_mapprof_separator)
                         for dattr in list_Data_attr]
    dict_data, new_kwargs = analyse_set_kwargs(list_Data_attr_up,
                                   separator=default_data_separator,
                                   **kwargs)
    return dict_data, new_kwargs

def analyse_maps_kwargs(reference_list=list_Map_attr, **kwargs):
    dict_maps, my_kwargs = analyse_set_kwargs(reference_list,
                                   separator=default_mapprof_separator,
                                   **kwargs)
    for map_name in dict_maps.keys():
        dict_data, new_kwargs = analyse_data_kwargs(map_name, **my_kwargs)
        for dmap_name, list_keywords in dict_data.items():
            for dmap_key in list_keywords:
                # Keyword with datamap name
                keydata = remove_suffix(dmap_key, map_name,
                                        separator=default_mapprof_separator)
                keyword = add_suffix(keydata, dmap_name,
                                     separator=default_data_separator)
                # Full keyword with map name AND datamap name
                full_keyword = add_suffix(dmap_key,
                                          dmap_name,
                                          separator=default_data_separator)

                dict_maps[map_name][keyword] = my_kwargs.pop(full_keyword, None)

    return dict_maps

def analyse_set_kwargs(given_attr, separator=default_mapprof_separator, **kwargs):
    """Analyse the kwargs arguments to detect names from maps or datamaps
    and send back a list of associated arguments for each dataset (Map, Profile,
    or DataMap or DataProfile).

    Args:
        given_attr (list of str): list of fixed attributes
        **kwargs: list of named arguments to analyse

    Returns:
        list_kwargs (list): list of dictionaries of all map names and their
            arguments
    """
    # Getting all the Map names and their arguments
    # This is using a list of given attribues which provides the list
    # of expected arguments for these
    dict_set_suffix = extract_suff_from_keywords(kwargs.keys(),
                                                     given_attr)

    dict_set_kwargs = {}
    # Loop over the suffix (default mname) of each Map
    for suffix_set in dict_set_suffix.keys():
        # Initialise the kwargs for Map
        set_kwargs = {}
        # First go through all the Map specific keyword and get the kwargs value
        for key in dict_set_suffix[suffix_set]:
            set_kwargs[key] = kwargs.pop(add_suffix(key, suffix_set,
                                                    separator=""), None)

        # If mname was not defined, we use the suffix_set as default
        # Assuming suffix_set is not empty. Otherwise we set a dummy.
        suffix_set = "{}".format(suffix_set.replace(separator, "", 1)
                                 if suffix_set.startswith(separator) else suffix_set)
        dict_set_kwargs[suffix_set] = set_kwargs

    return dict_set_kwargs, kwargs

def extract_suff_from_keywords(keywords, given_arglist, separator=""):
    """Extract list of keywords starting with a suffix
    assuming they all start with a string belonging to
    a given list of args.

    Attributes
        keywords: list of str
            Input keywords to analyse
        given_arglist: list of str
            Fixed list of arguments to test against
        separator: str [""]
            Letter or string used to separate items

    Returns
        Dictionary with suffixes as keys and list of found
        args as values
    """
    # Final dictionary - initialise
    dict_kwarg = {}

    # dictionary of ambiguous keywords (starting with the same prefix)
    dict_doublet = {}
    for arg in given_arglist:
        for arg2 in given_arglist:
            # If arg is in arg2 but they are different
            # There is a risk that it brings confusion
            # Hence include them in the doublet dict
            if arg in arg2 and arg != arg2:
                # if arg is already in the dict, just append the second item
                if arg in dict_doublet.keys():
                    dict_doublet[arg].append(arg2)
                # Otherwise start the list
                else:
                    dict_doublet[arg] = [arg2]

    # Now start with the given argument list
    for arg in given_arglist:
        # If there is confusion, then use that list
        if arg in dict_doublet.keys():
            larg2 = dict_doublet[arg]
        else:
            larg2 = ["###"]
        # We look for the keywords which starts with one of the arg
        found_suffixes = []
        for key in keywords:
            if key is None:
                continue
            # If arg is really the right prefix
            if key.startswith(arg) and not key.startswith(tuple(larg2)):
                # if key=arg, it means that the suffix should be empty ""
                if key == arg:
                    found_suffixes.append(key.replace(arg, ""))
                # Otherwise use the separator
                else:
                    found_suffixes.append(key.replace(arg + separator, ""))
        # For all of these, we extract the arg and add it to the dictionary
        for suffix in found_suffixes:
            if suffix in dict_kwarg.keys():
                dict_kwarg[suffix].append(arg)
            else:
                dict_kwarg[suffix] = [arg]

    return dict_kwarg

def extract_pref_from_keywords(keywords, given_arglist):
    """Extract list of keywords ending with a prefix
    assuming they all end with a string belonging to
    a given list of args

    Attributes
        keywords: list of str
            Keywords to test
        given_arglist: list of str
            Fixed list of args

    Returns
        Dictionary with prefixes as keys and list of found
        args as values
    """
    dict_kwarg = {}
    for arg in given_arglist:
        # We look for the keywords which starts with one of the arg
        found_prefixes = [kwarg.replace(arg, "") for key in keywords if key.endswith(arg)]
        # For all of these, we extract the arg and add it to the dictionary
        for prefix in found_prefixes:
            if prefix in dict_kwarg.keys():
                dict_kwarg[prefix].append(arg)
            else:
                dict_kwarg[prefix] = [arg]

    return dict_kwarg

