# -*- coding: utf-8 -*-
"""
A small module to check consistencies and input values
"""
import numpy as np

__author__ = "Eric Emsellem"
__copyright__ = "Eric Emsellem"
__license__ = "mit"

# -----------------------------------------------------------
# First, a set of useful function to check consistency
# -----------------------------------------------------------
# Check if 1D and all sizes are consistent
def _check_sizes_and_flatten(list_arrays) :
    """Check if all arrays in list have the same size
    and are 1D. If the arrays are nD, there are all flattened.

    Parameters:
    -----------
    list_arrays: list of numpy arrays

    Return
    ------
    Boolean and input list of (flattened!) arrays
    """
    Testok = True
    # Check formats and sizes
    if not _check_ifarrays(list_arrays) :
        print("ERROR: not all input data are arrays")
        Testok = False
    for i in range(len(list_arrays)) :
        if np.ndim(list_arrays[i]) > 1 :
            list_arrays[i] = list_arrays[i].ravel()
    if not _check_consistency_sizes(list_arrays) :
        print("ERROR: not all input data have the same length")
        Testok = False
    if not _check_ifnD(list_arrays) :
        print("ERROR: not all input data are 1D arrays")
        Testok = False

    return Testok, list_arrays

# Check if all sizes are consistent
def _check_consistency_sizes(list_arrays) :
    """Check if all arrays in list have the same size

    Parameters:
    -----------
    list_arrays: list of numpy arrays

    Return
    ------
    bool: True if all arrays are consistent, False otherwise.
    """
    if len(list_arrays) == 0 :
        return True

    return all(myarray.size == list_arrays[0].size for myarray in list_arrays)


# Check if all are arrays
def _check_ifarrays(list_arrays) :
    """Check if all items in the list are numpy arrays

    Parameters:
    -----------
    list_arrays: list of numpy arrays

    Return
    ------
    bool: True if all are arrays, False otherwise.
    """
    if len(list_arrays) == 0 :
        return True

    return all(isinstance(myarray, (np.ndarray)) for myarray in list_arrays)


# Check if all are nD arrays
def _check_ifnD(list_arrays, ndim=1) :
    """Check if all are of a certain dimension

    Parameters:
    -----------
    list_arrays: list of numpy arrays
    ndim: dimension which is expected (integer, default is 1)

    Return
    ------
    bool: True if all are "n"D, False otherwise.
    """
    if len(list_arrays) == 0 :
        return True

    return all(np.ndim(myarray) == ndim for myarray in list_arrays)
