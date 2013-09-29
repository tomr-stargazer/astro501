"""
Code relating to Problem 2 in HW #1a for Astro 501.

Problem 2 is about Optics.

See http://scipy-lectures.github.io/advanced/mathematical_optimization/
if you get scared about optimization.

Also, http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.optimize.brute.html

Refractive indices come from here:
http://refractiveindex.info/?group=GLASSES&material=BK7
http://refractiveindex.info/?group=SCHOTT&material=N-F2

Unrelatedly, check out the cubehelix color palette:
http://www.ifweassume.com/2013/05/cubehelix-or-how-i-learned-to-love.html

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brute

# n(lambda) for BK7
n1 = lambda wavelength: 1.541316168 - 0.0418*wavelength
# n(lambda) for Schott F2
n2 = lambda wavelength: 1.6706 - 0.0862*wavelength

def problem_2b():
    """
    Makes a plot comparing n_1(lambda) to n_2(lambda).

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure we plot the solution of 2b onto.

    """

    wavelength_range = np.arange(0.4, 0.85, 0.05)

    fig = plt.figure()

    plt.plot( wavelength_range, n2(wavelength_range), lw=2, label="Schott F2")
    plt.plot( wavelength_range, n1(wavelength_range), lw=2, label="BK7")

    plt.legend()

    plt.ylim(1.45, 1.7)
    plt.ylabel(r"Index of refraction $n(\lambda)$")
    plt.xlabel(r"Wavelength $\lambda (\mu m)$")

    plt.suptitle("Astro 501, Homework #1a, Problem 2b. Tom Rice")
    plt.title("Refractive indices of two materials versus wavelength.")

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
    """ 
    Gives the compund lens focal length at each wavelength.

    Parameters
    ----------
    lens_curvatures : tuple of 2 floats
        The values (in mm) for R1 and R2, respectively.
    wavelengths : np.ndarray of floats
        An array of wavelengths for which to calculate
        the focal length.

    Returns
    -------
    fc : np.ndarray of floats
        The focal distance of the compound lens per wavelength.
        This array is the same length as `wavelengths`.
    
    """
    
    R1, R2 = lens_curvatures

    # These two lines' math is the part of this code I am most shaky about.
    left_fraction = -(R2/(2 * (n2(wavelengths) - 1)))**(-1)
    right_fraction = (s - R1/(2*(n1(wavelengths) - 1)))**(-1)

    inverse_fc = left_fraction - right_fraction

    fc = 1 / inverse_fc
    
    return fc

def rms_deviation_over_all_wavelengths(lens_curvatures, wavelengths,
                                       target_focal_length=200, 
                                       func=fc_per_wavelength):
    """
    Gives the rms deviation, over all provided wavelengths,
    of the actual focal length from the target focal length.

    Parameters
    ----------
    lens_curvatures : tuple of 2 floats
        The values (in mm) for R1 and R2, respectively.
    wavelengths : np.ndarray of floats
        An array of wavelengths for which to calculate
        the rms values.
    target_focal_length : float, optional, default: 200
        The desired focal length, in mm, that you want to 
        calculate deviations from. Defaults to 200 mm because
        that's what John Monnier said to use on the homework.
    func : function, optional, default: fc_per_wavelength
        Which function to use for the 'actual' fc value when
        subtracting 'expected' minus 'actual'. Use the default.
    
    Returns
    -------
    rms : np.ndarray of floats
        An array of root-mean-square deviations from the target 
        focal length. This array is the same length as `wavelengths`.

    """

    deviation_array = target_focal_length - func(lens_curvatures, wavelengths)

    rms = np.sqrt(np.mean(deviation_array**2))

    return rms



def problem_2c(optimize_function=brute, **kwargs):
    """
    Finds the values for R1, R2 that optimize fc_per_wavelength around 200mm.

    It's general -- it takes in different optimize functions depending on your
    mood of the day, and passes extra keyword arguments into them.

    Parameters
    ----------
    optimize_function : function, optional, default: brute
        Which scipy.optimize function to use as an optimizer.
        Now defaults to `brute` so that we can plot the output using
        plt.imshow.

    Returns
    -------
    result : output of optimize_function
        Returns the output of optimize_function. Look at the
        documentation for whatever you're using as optimize_function.

    """

    if 'guess' not in kwargs:
        if optimize_function == brute:
            guess = (slice(5,250,2),slice(5,500,2))
        elif optimize_function == minimize:
            guess=(10,10) 
        else:
            guess=(100,100)
    else:
        guess=kwargs['guess']
        del kwargs['guess']

    wavelength_array =  np.arange(0.4, 0.8, 0.001)

    return optimize_function(
        lambda lens_curvatures: \
        rms_deviation_over_all_wavelengths(lens_curvatures, wavelength_array),
        guess, **kwargs)

def plot_solution_space_contours(**kwargs):
    """
    Makes a contour plot showing the best values for R1, R2.

    Calls problem_2c using brute and full_output=True, and then works
    with the information from that.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure we plot the contours onto.    
    
    """

    ((R1, R2), residual,
     input_grid, output_grid) = problem_2c(full_output=True)

    fig = plt.figure()

    # We do a transpose because imshow has a strange default order.
    img = plt.imshow(output_grid.transpose(), origin='lower',
                     vmin=0, vmax=100, cmap='cubehelix', 
                     extent=(
                         input_grid[0].min(), input_grid[0].max(), 
                         input_grid[1].min(), input_grid[1].max()
                         )
                    )
    cbar = plt.colorbar(img)

    cbar.set_label("Total rms of fit")

    plt.xlabel(r"$R_1$ (mm)")
    plt.ylabel(r"$R_2$ (mm)")

    plt.suptitle("Astro 501, Homework #1a, Problem 2c. Tom Rice")
    plt.title("How good is the fit as a function of $R_1$ and $R_2$?")

    plt.plot([R1], [R2], "r+", ms=75, markeredgewidth=2)

    plt.text(R1+20, R2-70,
             ("Best fit:\n\n"
              "$R_1 = %.3f$ mm,\n"
              "$R_2 = %.3f$ mm,\n\n"
              "Residual rms: %.4f mm" % (R1, R2, residual)))


    return fig

def plot_solution(lens_curvatures, wavelength_array):
    """ 
    Plots the rms deviation for the given lens_curvatures vs wavelength.
    
    Parameters
    ----------
    lens_curvatures : tuple of 2 floats
        The values (in mm) for R1 and R2, respectively.
    wavelengths : np.ndarray of floats
        An array of wavelengths for which to calculate
        the rms values.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure we plot the line onto.
    
    """

    fig = plt.figure()

    R1 = lens_curvatures[0]
    R2 = lens_curvatures[1]

    plt.plot(wavelength_array, 
             fc_per_wavelength(lens_curvatures, wavelength_array))

    plt.title(r"Focal length solution: $R_1$=%.2f mm and $R_2$=%.2f mm" % 
              (R1, R2))
    plt.xlabel(r"Wavelength $\lambda (\mu m)$")
    plt.ylabel("Back focal distance of compound lens")

    plt.show()

    return fig
    
    
