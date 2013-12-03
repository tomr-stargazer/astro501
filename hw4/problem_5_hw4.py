"""
Code to compute and plot Problem 5b in HW #4.

Limiting magnitude for an optical interferometer as a function
of wavelength.

"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from astropy.constants import k_B, c, R_sun, h
from astropy.units.quantity import Quantity

v_wind = Quantity(20, unit='m/s')
r_o = Quantity(20, unit='cm')

nu_550 = (c / Quantity(550, unit='nm')).to('Hz')

R_vega = 2.5 * R_sun
D_vega = Quantity(2.36e17, unit='m')
T_vega = Quantity(10000, unit='K')


delta_t = Quantity(1, unit='cm')/c

def nu_from_lambda(lamda):
    """ Computes frequency from wavelength in nm. """
    return c / Quantity(lamda, unit='nm')

def limiting_magnitude(nu):

    nu = Quantity(nu, unit='Hz')
    
    delta_nu = nu/10

    F_limiting = (20*h*v_wind*delta_nu)/(
        delta_t*nu*(r_o*(nu_550/nu).value**(6/5))**3)
    print F_limiting.to('erg cm-2 s-1').unit
    
    F_vega = 2*np.pi*(R_vega**2 *delta_nu * 2*h*nu**3)/(
        D_vega**2 * c**2 * (np.exp(h*nu/(k_B*T_vega))-1))
    print F_vega.to('erg cm-2 s-1').unit

    m_limiting = -2.5 * np.log10(F_limiting.to('erg cm-2 s-1')/
                                 F_vega.to('erg cm-2 s-1'))

    return m_limiting
