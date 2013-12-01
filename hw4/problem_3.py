"""
Code to solve problem 3 in Homework #4 for Astro 501.

This problem involves figuring out the angular size of a star
given a visibility and a projected baseline, assuming the star
is a uniform disk.

Part a: V**2 = 0.75 \pm 0.05
Part b: V**2 = 0.10 \pm 0.03

"""


def uniform_disk_visibility(angular_size, baseline_separation):
    """
    The visibility function V of a uniform disk, such as a star.

    Parameters
    ----------
    angular_size : float
        The size in radians of the star's disk
    baseline_separation : float
        The projected baseline, scaled by the wavelength.

    Returns
    -------
    visibility : float
        The visibility V(s).

    """
