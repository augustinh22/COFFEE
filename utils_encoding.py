#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 RSGB/UniBe (Remote Sensing Research Group, University of Bern)
#		developed for SemantiX Project (in cooperation with ZGIS, University of Salzburg)
#
# This utils_encoding code is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# utils_encoding is distributed in the hope that it will be useful, but without any warranty;
# without even the implied warranty of merchantability or fittness for a particular purpose.
# See the GNU General Public License for more details, here <http://www.gnu.org/licenses/>
#
# File:         utils_encoding.py
#
# Author: Helga Weber, RSGB/UniBe (Date: 2022/04/26; update: 2022/08/18)
import numpy as np
import xarray as xr

def unpackbits(data_array, num_bits):
    """Unpack bits for large numeric numpy arrays (i.e. any dimension ndarray).
    Ndarray must be int-like.

    Parameters
    ----------
    x : array_like
        The int-like numpy data array.
    num_bits : int
        Number of bits to unpack.

    Returns
    -------
    x & mask : array_like
        N-dimensional data array of unpacked bits.

    Source: answer 16:
    https://stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types
    Author: see source, documentation by Helga Weber RSGB/Unibe, 20220422

    """
    if np.issubdtype(data_array.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(data_array.shape)
    data_array = data_array.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=data_array.dtype).reshape([1, num_bits])
    return (data_array & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def get_encoded_bits(scene, var_name, num_bits):
    """Get encoded bits of scene or image.
    Unpack bits of xarray scene or image by using a data variable's name
    and number of bits for unpacking.
    Requires: function unpackbits

    Parameters
    ----------
    scene : array-like
	Data array as xarray.
    var_name : str
	Name of xarray data variable.
    num_bits : int
	Number of bits to unpack.

    Returns
    -------
    scene_encoded : array-like
        N-dimensional data array of encoded bits, last dimension equals size of number of bits.

    Example: cloud_unpacked = get_encoded_bits(scn, var_name='cloud', num_bits=16)

    Author: Helga Weber, RSGB/UniBe, 20220422

    """
    scene_encoded = unpackbits(scene[var_name].values, num_bits)
    return scene_encoded


def get_encoded_bands_as_dataset(scene_encoded, var, bit_names_encoded, num_bits):
    """Return encoded scene variables (bands) as xarray dataset.
    Works for 4D arrays.

    Parameters
    ----------
    scene_encoded : array-like
	Data array of encoded bits.
    var : int
	Data variable.
    bit_names_encoded : list of str
	List of strings, corresponding to names for encoded variables.
    num_bits : int
	Number of encoded bits.

    Returns
    -------
    combined_dataset : array-like
	Xarray data set of encoded bands as data variables.

    Example: cloud_ds = get_encoded_bands_as_dataset(
                            cloud_unpacked, scn['cloud'], cloud_names, num_bits=16
                        )

    Author: Helga Weber, RSGB/Unibe, 20220422

    """
    datasets = []
    for i in range(num_bits):
        data_var = xr.DataArray(scene_encoded[:,:,:,i], coords=var.coords, dims=var.dims)
        data_var.name = bit_names_encoded[i]
        datasets.append(data_var.to_dataset())
    combined_dataset = xr.merge(datasets)
    return combined_dataset


def unpack_slstr_clouds(scene_path):
    """Return encoded scene SLSTR confidence and cloud variables
    (bands) as xarray dataset. Works for 4D arrays.

    Parameters
    ----------
    scene_path : str
	Filepath to single file containing SLSTR encoded variables
    (e.g. NetCDF file containing reflectances, geometry and encoded variables)

    Returns
    -------
    cloud_ds : array-like
	Xarray data set of encoded bands as data variables.

    Author: Helga Weber, RSGB/Unibe, and Hannah Augustin, PLUS, 20220819

    """
    cloud_names = [
        'cloud_test1', 'cloud_test2', 'cloud_test3', 'cloud_test4',
        'cloud_test5', 'cloud_test6', 'cloud_test7', 'cloud_test8',
        'cloud_test9', 'cloud_test10', 'cloud_test11', 'cloud_test12',
        'cloud_test13', 'cloud_test14', 'cloud_test15', 'cloud_test16'
    ]
    confidence_names = [
        'confidence_coastline', 'confidence_ocean', 'confidence_tidal',
        'confidence_land', 'confidence_inland_water', 'confidence_unfilled',
        'confidence_spare1', 'confidence_spare2', 'confidence_cosmetic',
        'confidence_duplicate', 'confidence_day', 'confidence_twilight',
        'confidence_sun_glint', 'confidence_snow', 'confidence_summary_cloud',
        'confidence_summary_pointing'
    ]
    #
    # Open dataset
    # Instead of opening a file from a filepath, you could pass the xarray object
    # containing the encoded arrays.
    #
    scn = xr.open_dataset(scene_path)

    confidence_unpacked = get_encoded_bits(scn, var_name='confidence', num_bits=16)
    cloud_unpacked = get_encoded_bits(scn, var_name='cloud', num_bits=16)

    confidence_ds = get_encoded_bands_as_dataset(
        confidence_unpacked, scn['confidence'], confidence_names, num_bits=16
    )
    cloud_ds = get_encoded_bands_as_dataset(
        cloud_unpacked, scn['cloud'], cloud_names, num_bits=16
    )

    scn_ds = xr.merge((confidence_ds, cloud_ds))
    scn_ds = scn_ds.astype('byte')

    return scn_ds
