"""
Code to solve problem 3 in Homework #4 for Astro 501.

This problem involves figuring out the angular size of a star
given a visibility and a projected baseline, assuming the star
is a uniform disk.

Part a: V**2 = 0.75 \pm 0.05
Part b: V**2 = 0.10 \pm 0.03

I use scipy.optimize.minimize_scalar, which minimizes a single-variable
function within a given range.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv
from scipy.optimize import minimize_scalar


def uniform_disk_visibility(angular_size, baseline_separation):
    """
    The visibility function V of a uniform disk, such as a star.

    Parameters
    ----------
    angular_size : float
        The size in radians of the star's disk (a)
    baseline_separation : float
        The projected baseline, scaled by the wavelength. (s)

    Returns
    -------
    visibility : float
        The visibility V(s).

    """

    a = angular_size
    s = baseline_separation

    visibility = np.abs(2 * jv(1, np.pi * a * s) / (np.pi * a * s))
    
    return visibility

def plot_initial_guess_of_a(baseline_separation, visibility_squared):
    """
    Makes a plot that lets you eyeball a first guess at a solution for `a`.

    Parameters
    ----------
    baseline_separation : float
        The projected baseline, scaled by the wavelength. (s)    
    visibility_squared : float
        The measured visibility squared V**2.

    Returns
    -------
    fig : figure
        Figure that we plot the initial guess onto.

    """

    fig = plt.figure()

    # guess array is in radians. This range should work.
    guess_array = np.arange(1e-9, 5e-9, 1e-11)
    s = baseline_separation

    plt.plot( guess_array, np.abs(
        uniform_disk_visibility(guess_array, s)**2 - visibility_squared) )

    return fig


def solve_problem_3(baseline_separation, visibility_squared,
                    output_milliarcsec=True):
    """
    Solves for `a` given a baseline separation and a visibility-squared.

    The solution range was informed by plot_initial_guess_of_a().

    Parameters
    ----------
    baseline_separation : float
        The projected baseline, scaled by the wavelength. (s)    
    visibility_squared : float
        The measured visibility squared V**2.

    Returns
    -------
    angular_size : float
        The optimal angular size of the star given the inputs.
    
    """

    solution_range = [1e-9, 5e-9] # in radians
    s = baseline_separation

    visibility_function = lambda a: np.abs(
        uniform_disk_visibility(a, s)**2 - visibility_squared)

    solution = minimize_scalar(visibility_function, solution_range,
                               tol=0.001)

    if output_milliarcsec:
        solution.x = np.degrees(solution.x)*3600*1000

    if solution.fun < 0.002:
        return solution.x
    else:
        raise ValueError("Function failed to converge within the given range.")

