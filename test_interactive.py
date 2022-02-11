import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from IPython.display import clear_output
import time
from pathlib import Path

import proper
import roman_phasec_proper
roman_phasec_proper.copy_here()

import time

for i in range(60): 
    print(i)
    time.sleep(1)

hdu = fits.PrimaryHDU(data=np.zeros((100,100)))
hdu.writeto('test_interactive.fits', overwrite=True)
