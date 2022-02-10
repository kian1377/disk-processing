import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from IPython.display import clear_output
import time
from pathlib import Path

import proper
# proper.prop_use_fftw()
# proper.prop_fftw_wisdom( 1024 ) 

import roman_phasec_proper
roman_phasec_proper.copy_here()

import misc
from matplotlib.patches import Circle

# data_dir = Path('C:/Users/Kian/Documents/data-files/disk-processing')
# data_dir = Path('/users/kianmilani/Documents/data-files/disk-processing')
data_dir = Path('/groups/douglase/kians-data-files/disk-processing')

wavelength_c = 575e-9*u.m
D = 2.3631*u.m
mas_per_lamD = (wavelength_c/D*u.radian).to(u.mas)

pixelscale_m_ref = 13e-6*u.m/u.pixel
pixelscale_lamD_ref = 1/2
wavelength_ref = 0.5e-6*u.m

# define desired PSF dimensions and pixelscale in units of lambda/D
npsf = 256                   # output image dimension (must be power of 2)
psf_pixelscale_lamD = 0.1    # output sampling in lam0/D
psf_pixelscale_mas = psf_pixelscale_lamD*mas_per_lamD/u.pix

iwa = 2.8
owa = 9.7

# Define the polarization axis you want to create the psfs for
#    -2 = -45d in, Y out 
#    -1 = -45d in, X out 
#     1 = +45d in, X out 
#     2 = +45d in, Y out 
#     5 = mean of modes -1 & +1 (X channel polarizer)
#     6 = mean of modes -2 & +2 (Y channel polarizer)
#    10 = mean of all modes (no polarization filtering)
polaxis = 10




sampling1 = 0.05
offsets1 = np.arange(0,iwa+1,sampling1)
# print(offsets1)

sampling2 = 0.1
offsets2 = np.arange(iwa+1,owa,sampling2)
# print(offsets2)

sampling3 = 0.25
offsets3 = np.arange(owa,15+sampling3,sampling3)
# print(offsets3)

r_offsets = np.hstack([offsets1, offsets2, offsets3])
r_offsets_mas = r_offsets*mas_per_lamD
print(r_offsets.shape, r_offsets_mas)

sampling_theta = 12
thetas = np.arange(0,360,sampling_theta)*u.deg
print(thetas.shape, thetas)

psfs_required = len(thetas)*len(r_offsets)
psfs_size_gb = psfs_required*(256**2)*8/1e9
time_required = 17*len(thetas)*len(r_offsets)/3600
print(psfs_required, psfs_size_gb, time_required)


nlam = 1
lam0 = 0.575
if nlam==1:
    lam_array = np.array([lam0])
else:
    bandwidth = 0.1
    minlam = lam0 * (1 - bandwidth/2)
    maxlam = lam0 * (1 + bandwidth/2)
    lam_array = np.linspace( minlam, maxlam, nlam )

xoffset = 0
use_fpm = 1

use_hlc_dm_patterns = 0
use_errors = 1
use_dm1 = 1
use_dm2 = 1
dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir + r'/examples/hlc_best_contrast_dm1.fits' )
dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir + r'/examples/hlc_best_contrast_dm2.fits' )
use_fieldstop = 1
use_pupil_defocus = 1

(wfs, pxscls_m) = proper.prop_run_multi('roman_phasec', lam_array, npsf, QUIET=False, 
                                        PASSVALUE={'cor_type':'hlc', # change coronagraph type to correct band
                                                   'final_sampling_lam0':psf_pixelscale_lamD, 
                                                   'source_x_offset':xoffset,
                                                   'use_fpm':use_fpm,
                                                   'use_hlc_dm_patterns':use_hlc_dm_patterns,
                                                   'use_errors': use_errors,
                                                   'use_lens_errors':use_errors,
                                                   'use_dm1':use_dm1, 'dm1_m':dm1, 
                                                   'use_dm2':use_dm2, 'dm2_m':dm2,
                                                   'use_field_stop':use_fieldstop,
                                                   'polaxis':polaxis,
                                                  })

psfs = np.abs(wfs)**2
psf_bb = np.sum(psfs, axis=0)/nlam
psf_pixelscale_m = pxscls_m[0]*u.m/u.pix
misc.myimshow(psf_bb, lognorm=True, pxscl=psf_pixelscale_m.to(u.mm/u.pix))


band1_wavelength = 575e-9*u.m
iwa_band1 = 2.8
owa_band1 = 9.7

iwa = iwa_band1 * wavelength_c/band1_wavelength
owa = owa_band1 * wavelength_c/band1_wavelength

iwa_mas = iwa*mas_per_lamD
owa_mas = owa*mas_per_lamD

patches1 = [Circle((0, 0), iwa.value, color='c', fill=False), Circle((0, 0), owa.value, color='c', fill=False)]
patches2 = [Circle((0, 0), iwa_mas.value, color='c', fill=False), Circle((0, 0), owa_mas.value, color='c', fill=False)]
misc.myimshow2(psf_bb, psf_bb, lognorm1=True, lognorm2=True, 
               pxscl1=psf_pixelscale_lamD, pxscl2=psf_pixelscale_mas,
               patches1=patches1, patches2=patches2)



psfs_array = np.zeros( shape=( (len(r_offsets)-1)*len(thetas) + 1,npsf,npsf) )
print(psfs_array.shape)

start = time.time()
count = 0
for r in r_offsets: 
    for th in thetas:
        xoff = r*np.cos(th)
        yoff = r*np.sin(th)
        (wfs, pxscls_m) = proper.prop_run_multi('roman_phasec', lam_array, npsf, QUIET=True, 
                                            PASSVALUE={'cor_type':'hlc',
                                                       'final_sampling_lam0':psf_pixelscale_lamD, 
                                                       'source_x_offset':xoff.value,
                                                       'source_y_offset':yoff.value,
                                                       'use_fpm':use_fpm,
                                                       'use_hlc_dm_patterns':use_hlc_dm_patterns,
                                                       'use_errors':use_errors,
                                                       'use_lens_errors':use_errors,
                                                       'use_dm1':use_dm1, 'dm1_m':dm1, 
                                                       'use_dm2':use_dm2, 'dm2_m':dm2,
                                                       'use_field_stop':use_fieldstop,
                                                       'polaxis':polaxis,
                                                      })

        psfs = np.abs(wfs)**2
        psf = np.sum(psfs, axis=0)/nlam
        psf_pixelscale_m = pxscls_m[0]*u.m/u.pix
    
        psfs_array[count] = psf
        clear_output(wait=True)
        print('{:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(count, xoff, yoff, r, th), time.time()-start)
        count += 1
        if r<r_offsets[1]: break
            
            
            
hdr = fits.Header()
hdr['PXSCLAMD'] = psf_pixelscale_lamD
hdr.comments['PXSCLAMD'] = 'pixel scale in lam0/D per pixel'
hdr['PXSCLMAS'] = psf_pixelscale_mas.value
hdr.comments['PXSCLMAS'] = 'pixel scale in mas per pixel'
hdr['PIXELSCL'] = psf_pixelscale_m.value
hdr.comments['PIXELSCL'] = 'pixel scale in meters per pixel'
hdr['POLAXIS'] = polaxis
hdr.comments['POLAXIS'] = 'polaxis: defined by roman_phasec_proper'

psfs_hdu = fits.PrimaryHDU(data=psfs_array, header=hdr)

psfs_fpath = data_dir/'psfs'/'hlc_band1_polaxis{:d}_rtheta_2.fits'.format(polaxis)
psfs_hdu.writeto(psfs_fpath, overwrite=True)


