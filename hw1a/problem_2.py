"""
Code relating to Problem 2 in HW #1a for Astro 501.

Problem 2 is about Optics.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

def problem_2b():
    """
    Makes a plot comparing n_1(lambda) to n_2(lambda).

    """

    # n(lambda) for BK7
    n1 = lambda lam: 1.541316168 - 0.0418*lam
    # n(lambda) for Schott F2
    n2 = lambda lam: 1.6706 - 0.0862*lam

    
