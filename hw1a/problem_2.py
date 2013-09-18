"""
Code relating to Problem 2 in HW #1a for Astro 501.

Problem 2 is about Optics.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# n(lambda) for BK7
n1 = lambda wavelength: 1.541316168 - 0.0418*wavelength
# n(lambda) for Schott F2
n2 = lambda wavelength: 1.6706 - 0.0862*wavelength

def problem_2b():
    """
    Makes a plot comparing n_1(lambda) to n_2(lambda).

    """

    wavelength_range = np.arange(0.4, 0.85, 0.05)

    fig = plt.figure()

    plt.plot( wavelength_range, n2(wavelength_range), lw=2, label="Schott F2")
    plt.plot( wavelength_range, n1(wavelength_range), lw=2, label="BK7")

    plt.legend()

    plt.ylim(1.45, 1.7)
    plt.ylabel(r"Index of refraction $n(\lambda)$")
    plt.xlabel(r"Wavelength $\lambda (\mu m)$")

    plt.title("Problem 2b")

    plt.text(0.5, 1.63, r"$n_2(\lambda) = 1.6707 - 0.0862 \lambda/\mu m$", 
             fontsize=18)
    plt.text(0.5, 1.52, r"$n_1(\lambda) = 1.5414 - 0.0418 \lambda/\mu m$", 
             fontsize=18)
    
    plt.show()
    return fig

# All lengths in millimeters.

# s: distance between lenses
s = 1 

def fc_per_wavelength(lens_curvatures, wavelengths):
    """ Gives the inverse focal length at each wavelength."""
    R1 = lens_curvatures[0]
    R2 = lens_curvatures[1]

    # These two lines' math is the part of this code I am most shaky about.
    left_fraction = 2 * (n2(wavelengths) - 1)/R2
    right_fraction = (s - R1/(2*(n1(wavelengths) - 1)))**(-1)

    inverse_fc = left_fraction - right_fraction

    return 1/inverse_fc


def problem_2c(guess=(10,10)):
    """
    This is the minimizing problem.

    """


    wavelength_array =  np.arange(0.4, 0.8, 0.05)

    #now, we want to get the rms deviation from 200

    def rms_deviation_over_all_wavelengths(lens_curvatures, wavelengths,
                                           target_focal_length=200, 
                                           func=fc_per_wavelength):
        """
        Gives the deviation, over all provided wavelengths,
        of the focal length from the target focal length.

        """

        deviation_array = target_focal_length - func(lens_curvatures, wavelengths)

        rms = np.sqrt(np.mean(deviation_array**2))

        return rms

    return minimize(
        lambda lens_curvatures: \
        rms_deviation_over_all_wavelengths(lens_curvatures, wavelength_array),
        guess)

    
    
#    print rms_deviation_over_all_wavelengths(guess, wavelength_array)
        

    
