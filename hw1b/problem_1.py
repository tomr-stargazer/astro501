"""
Code relating to Problem 1c in HW #1b for Astro 501.
This code is hosted online at https://github.com/tomr-stargazer/astro501 .

Problem 1 is about Fourier Transforms.

Since FFTs are super confusing, especially with the changes in
coordinates and the required shifts and the imaginary parts everywhere,
here are some links I found helpful to learn how to do numpy FFTs properly:

http://stackoverflow.com/questions/5398304/fourier-transform-of-a-gaussian-is-not-a-gaussian-but-thats-wrong-python
http://stackoverflow.com/questions/11320312/numpy-scipy-fft-for-voltage-time-data
http://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html

(in other words, the official documentation plus Stack Overflow 
is incredibly helpful.)

This had a section on what a Gaussian aperture is:
http://www.uv.es/imaging3/lineas/apod.htm

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

def E_aperture(z, sigma_over_lambda=5):
    """
    Returns the value of the electric field in the aperture plane.

    Because the parameter $E_0$ is not mentioned in this problem,
    we are setting it to one, i.e. dividing it out of all of our
    expressions of E.

    The ratio $\sigma / \lambda$ is 5 by default, but is adjustable.
    I use the notation $z = x / \lambda$ in this function.

    Parameters
    ----------
    z : float 
        The coordinate along the aperture plane, scaled by lambda.
        Defined as $z = x / \lambda$.
    sigma_over_lambda : float, optional, default: 5
        The value of sigma divided by wavelength. John Monnier has 
        us using 5 in this problem, but it's fun to vary it.

    Returns
    -------
    E : float
        The strength of the electric field of the aperture plane
        at the coordinate z, scaled to the max strength $E_0$.

    """

    E = np.exp(-0.5 * z**2 / sigma_over_lambda**2)

    return E

def analytic_diffracted_power(theta, sigma_over_lambda=5):
    """
    Returns the analytically derived diffracted power.

    It's not normalized in any sense or to any units, so 
    you'll have to normalize it against that other diffracted power.

    Parameters
    ----------
    theta : float
        The input angular distance from the center of the beam.
    sigma_over_lambda : float, optional, default: 5
        The value of sigma divided by wavelength.

    Returns
    -------
    P : float
        The power of the diffracted electric field 
        at angle `theta`, in some arbitrary unit system that
        needs normalization.
    
    """

    P = np.exp(-4 * sigma_over_lambda**2 * np.pi**2 * theta**2)

    return P

def make_radial_profile_of_E_aperture(z_max=40, sigma_over_lambda=5):
    """
    Plots E versus $z = x / \lambda$ over some z range.

    Parameters
    ----------
    z_max : float, optional, default: 40
        The max value of z to plot on the x-axis.
    sigma_over_lambda : float, optional, default: 5
        The value of sigma divided by wavelength. This is
        passed onto E_aperture.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure we plot the solution onto.

    """
    
    if z_max < 0:
        raise ValueError("z_max must be a positive number!")

    z_array = np.arange(-z_max, z_max, 0.05)

    fig = plt.figure()

    plt.plot(z_array, E_aperture(z_array, 
                                 sigma_over_lambda=sigma_over_lambda))

    plt.suptitle("Astro 501, Homework #1b, Problem 1c. Tom Rice")
    plt.title("Radial profile of the aperture electric field, for "+
              "$\\sigma / \\lambda = %s $" % sigma_over_lambda)

    plt.xlabel(r"Distance (in units $ x / \lambda $) from aperture center" )

    plt.ylabel(r"$ E(x / \lambda) / E_0$")

    plt.show()
    return fig

def make_radial_profile_of_diffracted_power(z_range=1500,
                                            z_sampling=512,
                                            sigma_over_lambda=5,
                                            analytical_comparison=True):
    """
    Shows the radial profile of the squared Fourier transform of E.

    Parameters
    ----------
    z_range : int, optional, default: 1500
        What range of z values to use. A big number here means 
        a well-sampled function in Fourier space, so bigger really
        is better here.
    z_sampling : int, optional, default: 512
        How many incremental values of z to use in the calculation.
        This has no effect on the frequency/theta sampling, so don't
        go overboard here. You get bonus points if you use a power of 2.
    sigma_over_lambda : float, optional, default: 5
        The value of sigma divided by wavelength. This is
        passed onto E_aperture.
    analytical_comparison : bool, optional, default: True
        Compare the FWHM to an analytical prediction? Plots some text
        and some dotted lines over stuff.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure we plot the solution onto.

    """

    z_array = np.linspace(-z_range, z_range, z_sampling)
    E = E_aperture(z_array, sigma_over_lambda=sigma_over_lambda)

    # "Diffracted E" is the Fourier transform of E -- it's
    # the electric field as a function of angle.
    diffracted_E = np.fft.fftshift(np.fft.fft(E))
    diffracted_E_normalized = diffracted_E / diffracted_E.max()

    # Power is the magnitude of the electric field, squared.
    diffracted_power = np.abs(diffracted_E_normalized)**2

    # the theta values are like "spatial frequencies"
    theta_array = np.fft.fftshift( np.fft.fftfreq( z_array.size, 
                                                   d=z_array[1]-z_array[0]))

    fig = plt.figure()

    plt.plot(theta_array, diffracted_power, label="FFT-computed power")

    plt.ylabel("Power of diffracted field, normalized: "
               r"$|E(\theta)|^2 / E_0^2$")
    plt.xlabel(r"Diffracted angle $\theta$ (radians)")

    plt.suptitle("Astro 501, Homework #1b, Problem 1c. Tom Rice")
    plt.title("Radial profile of the diffracted power, for "+
              "$\\sigma / \\lambda = %s $" % sigma_over_lambda)

    plt.xlim(-0.3, 0.3)

    if analytical_comparison:

        power = analytic_diffracted_power(theta_array)
        plt.plot(theta_array, power / power.max(), scalex=False,
                 label="Analytic power")

        expected_fullwidth_halfmax =(np.sqrt(np.log(2)) /
                                     (sigma_over_lambda * np.pi))

        # This is a little hacky, but is the right way to do it
        # as long as the sampling is high enough
        actual_fullwidth_halfmax = (
            theta_array[diffracted_power >= 0.5].max() -
            theta_array[diffracted_power >= 0.5].min() )

        plt.plot([-expected_fullwidth_halfmax/2, 
                  expected_fullwidth_halfmax/2], 
                  [0.5, 0.5], 'r:', lw=3, label="Analytic FWHM prediction")

        plt.legend(loc="lower left")
        
        plt.text(-0.295, 0.5, 
                 ("Analytic calculation of FWHM:\n\n"
                  r"$\frac{\lambda \sqrt{ \ln 2 }}{\sigma \pi}$"
                  " = %.4f radians" % expected_fullwidth_halfmax
                  ),
                  fontsize=14)
        plt.text(0.03, 0.5,
                 ("Measured FWHM from this plot:\n\n"
                  r"$\theta_{FWHM}$ = %.4f radians" %
                  actual_fullwidth_halfmax),
                 fontsize=14)

    plt.show()
    return fig
