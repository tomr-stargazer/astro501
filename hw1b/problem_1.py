"""
Code relating to Problem 1c in HW #1b for Astro 501.

Problem 1 is about Fourier Transforms.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

# Part one: 
# Assuming sigma = 5 * lambda, use the computer to make a radial profile
# of the aperture electric field
# E(x) = E_0 exp( -0.5 * (x/sigma)^2)
# in units of z = x / lambda.

# Mathematically this gives
# E(z) = E_0 exp( -0.5 * (z/25) )

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
        Defined as $z = z / \lambda$.
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

    z_array = np.arange(0, z_max, 0.05)

    fig = plt.figure()

    plt.plot(z_array, E_aperture(z_array, 
                                 sigma_over_lambda=sigma_over_lambda))

    plt.suptitle("Astro 501, Homework #1b, Problem 1c. Tom Rice")
    plt.title("Radial profile of the aperture electric field, for "+
              "$\\sigma / \\lambda = %s $" % sigma_over_lambda)

    plt.xlabel(r" $ x / \lambda $ " )
    #    plt.ylabel(r"$ \frac{E(x / \lambda)}{ E_0 } $", rotation='horizontal')
    plt.ylabel(r"$ E(x / \lambda) / E_0$", 
               rotation='horizontal')

    plt.show()
    return fig

def make_radial_profile_of_diffracted_power(z_sampling=512, 
                                            sigma_over_lambda=5):
    """
    Shows the radial profile of the squared Fourier transform of E.

    Parameters
    ----------
    sigma_over_lambda : float, optional, default: 5
        The value of sigma divided by wavelength. This is
        passed onto E_aperture.

    """

    # Make an array of values from -20 to 20. 
    z_array = np.linspace(-20, 20, z_sampling)
    E = E_aperture(z_array, sigma_over_lambda=sigma_over_lambda)

    diffracted_E = np.fft.fftshift(np.fft.fft(E)) / np.sqrt(2*len(E))

    diffracted_power = np.abs(diffracted_E)**2

    # the theta values are like "spatial frequencies"
    theta_array = np.fft.fftshift( np.fft.fftfreq( z_array.size, 
                                                   d=z_array[1]-z_array[0]))

    fig = plt.figure()

    plt.plot(theta_array, diffracted_power)

    plt.ylabel(r" $|E(\theta)|^2 / E_0^2$")
    plt.xlabel(r"Diffracted angle $\theta$ (radians)")

    plt.title("Radial profile of the diffracted power, for "+
              "$\\sigma / \\lambda = %s $" % sigma_over_lambda)
    plt.show()
    return fig
