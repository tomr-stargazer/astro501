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

def E_aperture(z):
    """
    Returns the value of the electric field in the aperture plane.

    Because the parameter $E_0$ is not mentioned in this problem,
    we are setting it to one, i.e. dividing it out of all of our
    expressions of E.

    Also, we are assuming $\sigma = 5 * \lambda$, and I use the 
    notation $z = x / \lambda$ in this assumption.

    Parameters
    ----------
    z : float 
        The coordinate along the aperture plane, scaled by lambda.
        Defined as $z = z / \lambda$.

    Returns
    -------
    E : float
        The strength of the electric field of the aperture plane
        at the coordinate z, scaled to the max strength $E_0$.

    """

    E = np.exp(-0.5 * z**2 / 25)

    return E

def make_radial_profile_of_E_aperture(z_max=40):
    """
    Plots E versus $z = x / \lambda$ over some z range.

    Parameters
    ----------
    z_max : float, optional, default: 40
        The max value of z to plot on the x-axis.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure we plot the solution onto.

    """
    
    if z_max < 0:
        raise ValueError("z_max must be a positive number!")

    z_array = np.arange(0, z_max, 0.05)

    fig = plt.figure()

    plt.plot(z_array, E_aperture(z_array))

    plt.suptitle("Astro 501, Homework #1b, Problem 1c. Tom Rice")    
    plt.title("Radial profile of the aperture electric field")

    plt.xlabel(r" $ x / \lambda $ " )
    #    plt.ylabel(r"$ \frac{E(x / \lambda)}{ E_0 } $", rotation='horizontal')
    plt.ylabel(r"$ E(x / \lambda) / E_0$", 
               rotation='horizontal')

    plt.show()
    return fig

def make_radial_profile_of_diffracted_power(z_sampling):
    """
    Shows the radial profile of the squared Fourier transform of E.

    """

    # Make a really well-sampled array of values from -20 to 20.
    z_array = z = np.linspace(-20, 20, z_sampling)
    E = E_aperture(z)

    diffracted_E = np.fft.fftshift(np.fft.fft(E)) / np.sqrt(2*len(E))

    diffracted_power = np.abs(diffracted_E)**2

    # the theta values are like "spatial frequencies"
    theta_array = np.fft.fftshift( np.fft.fftfreq( z.size, d=z[1]-z[0]))

    fig = plt.figure()

    plt.plot(theta_array, diffracted_power)

    plt.ylabel(r" $|E(\theta)|^2 / E_0^2$")
    plt.xlabel(r"Diffracted angle $\theta$ (radians)")

    plt.show()
    return fig
