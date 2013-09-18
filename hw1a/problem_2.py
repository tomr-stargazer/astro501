"""
Code relating to Problem 2 in HW #1a for Astro 501.

Problem 2 is about Optics.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# n(lambda) for BK7
n1 = lambda lam: 1.541316168 - 0.0418*lam
# n(lambda) for Schott F2
n2 = lambda lam: 1.6706 - 0.0862*lam

def problem_2b():
    """
    Makes a plot comparing n_1(lambda) to n_2(lambda).

    """

    lam_range = np.arange(0.4, 0.85, 0.05)

    fig = plt.figure()

    plt.plot( lam_range, n2(lam_range), lw=2, label="Schott F2")
    plt.plot( lam_range, n1(lam_range), lw=2, label="BK7")

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

def problem_2c(guess=(10,10)):
    """
    This is the minimizing problem.

    """

    # All lengths in millimeters.
    
    s = 1 

    def inverse_fc_per_wavelength(r1r2, lam):
        r1 = r1r2[0]
        r2 = r1r2[1]

        # These two lines' math is the part of this code I am most shaky about.
        left_fraction = 2 * (n2(lam) - 1)/r2
        right_fraction = (s - r1/(2*(n1(lam) - 1)))**(-1)

        inverse_fc = left_fraction - right_fraction

        return inverse_fc

    lam_array =  np.arange(0.4, 0.85, 0.05)

    print 1/inverse_fc_per_wavelength(guess, lam_array)

    
