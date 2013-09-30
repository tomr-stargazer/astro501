"""
Code relating to Problem 1c in HW #1b for Astro 501.

Problem 1 is about Fourier Transforms.

"""

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
        at the coordinate z.

    """
