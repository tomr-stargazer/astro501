"""
Working on a project for Astro 501, which involves spectra.

# stuff!
http://mthamilton.ucolick.org/techdocs/instruments/nickel_spect/arcSpectra/

# ds9 tip
http://casa.colorado.edu/~ginsbura/ds9tips.htm#body1

"""

from __future__ import division

import numpy as np

wavelength_solution_array = {
    823: 5852.49,
    738: 5944.83,
    597: 6096.16,
    551: 6143.06,
    434: 6266.50,
    369: 6334.43,
    322: 6383.00,
    306: 6402.25,
    205: 6506.53,
    118: 6598.95 
    }

print wavelength_solution_array

w = wavelength_solution_array

a, b, c, d = np.polyfit(w.keys(), w.values(), 3)

wavelength_solution =  lambda x: a*x**3 + b*x**2 +c*x + d


