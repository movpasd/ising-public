"""Determining thermodynamic properties from Ising model simulation"""

import numpy as np


def magnetisation(a):
    """
    Calculate mean magnetisation

    a: (..., Nx, Ny)-array
    RETURNS: (...,)-array
    """

    return np.mean(a, axis=(-1, -2))


def square_mag(a):
    """
    Convenience function to calculate the square of magnetisation
    """

    return np.mean(a, axis=(-1, -2))**2


def energy(a, h=0):
    """
    Calculate energy of a grid

    Takes a (..., Nx, Ny)-array
    Returns a (...,)-array
    """

    # Each nearest-neighbour pair is either bottom-top or left-right.
    # To sum over every possible nearest-neighbour pair, generate
    # all bottom-top pairs and left-right pairs separately and add
    b = np.roll(a, 1, axis=-1)
    c = np.roll(a, 1, axis=-2)

    return -np.sum((b + c + h) * a, axis=(-1, -2))


def autocovariance(samples, maxtau=None, axis=-1, rem_dc=True):
    """
    Calculate the auto-correlation of a sampled function of time

    samples: float (iternum,)-array
    maxtau: int <= iternum
    rem_dc: bool -- whether to remove the DC component of the sampling
    """

    iternum = samples.shape[axis]

    if maxtau is None:
        maxtau = samples // 2
    else:
        assert type(maxtau) is int and maxtau <= iternum

    if rem_dc:
        samples = np.copy(samples - np.mean(samples, axis=axis))

    autoc_values = []
    for tau in range(maxtau):

        # Annoyingly, there's no way to pick an arbitrary axis using
        # slice notation so I have to use np.take

        # This is equivalent to s1 = samples[:,:,:-tau,:]; s2 = ..
        # with :'s in every axis but one
        s1 = np.take(samples, range(0, iternum - tau), axis=axis)
        s2 = np.take(samples, range(tau, iternum), axis=axis)
        autoc_values.append(np.mean(s1 * s2, axis=axis))

    return np.stack(autoc_values, axis=axis)


def autocorrelation(samples, maxtau=None, axis=-1):

    autocov = autocovariance(samples, maxtau, axis)
    return autocov / np.take(autocov, [0], axis=axis)


def rolling_average(values, window, axis=-1):
    """Take rolling average of array over axis"""

    # This solution is a bit messy because np.convolve only likes
    # 1d arrays and because slicing is messy if you need to take
    # values over an arbitrary axis

    # Basic idea is: the difference in the cumulated sum before and
    # after the window divided by the window is the average

    cs = np.cumsum(values, axis=axis)
    l = values.shape[axis]
    cs1 = np.take(cs, range(0, l - window), axis=axis)
    cs2 = np.take(cs, range(window, l), axis=axis)

    return (cs2 - cs1) / window


def isflat(testfunc, ensemble, timescale, tolerance, absolute=True):
    """
    Determines when and if a test function is ~constant over time
    over an ensemble

    testfunc: callable -- must take a microstate
    ensemble: datagen.Ensemble
    tolerance: float
    absolute: bool -- whether to interpret tolerance as absolute or relative
        defaults to True, i.e.: absolute tolerance

    RETURNS: (ensemble.iternum, - timescale)-array
    """

    iternum = ensemble.iternum

    # First, we calculate the ensemble average of the quantity
    # as it varies over time
    ens_avgs = np.array(ensemble.ensemble_avg(testfunc))

    # Next, we look at variations in that ensemble average over time
    diffs = np.diff(ens_avgs)

    # Take a rolling average of the changes over the timescale
    smoothed_diffs = rolling_average(diffs, timescale)

    # If the smoothed difference is less than the tolerance,
    # the test function can be said to be ~constant.
    #
    # i.e.: if
    if absolute:
        isflats = smoothed_diffs < tolerance
    else:
        smoothed_avgs = rolling_average(ens_avgs, timescale)
        isflats = smoothed_diffs / smoothed_avgs < tolerance

    return isflats
