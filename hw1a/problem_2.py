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
