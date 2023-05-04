import numpy as np
import astropy.io.fits as fits
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from IPython.display import clear_output, display
import time
from pathlib import Path
import copy

import proper
proper.prop_use_fftw(DISABLE=False)

import roman_phasec_proper

import cgi_phasec_poppy as cgi

import ray

import misc_funs as misc

data_dir = Path('/groups/douglase/kians-data-files/disk-processing')

dm1_best = fits.getdata(roman_phasec_proper.lib_dir + r'/examples/hlc_best_contrast_dm1.fits')
dm2_best = fits.getdata(roman_phasec_proper.lib_dir + r'/examples/hlc_best_contrast_dm2.fits')


wavelength_c = 575e-9*u.m
D = 2.3631*u.m
mas_per_lamD = (wavelength_c/D*u.radian).to(u.mas)


npsf = 64
psf_pixelscale = 13e-6
psf_pixelscale_lamD = 500/575 * 1/2
psf_pixelscale_mas = psf_pixelscale_lamD*mas_per_lamD/u.pix

iwa = 3
owa = 9

# Create the sampling grid the PSFs will be made on
sampling1 = 0.05
sampling2 = 0.1
sampling3 = 0.25
offsets1 = np.arange(0,iwa+1,sampling1)
offsets2 = np.arange(iwa+1,owa,sampling2)
offsets3 = np.arange(owa,15+sampling3,sampling3)

r_offsets = np.hstack([offsets1, offsets2, offsets3])
nr = len(r_offsets)
r_offsets_mas = r_offsets*mas_per_lamD
display(nr, r_offsets)

sampling_theta = 6
thetas = np.arange(0,360,sampling_theta)*u.deg
nth = len(thetas)
display(nth, thetas)

psfs_required = (nr-1)*nth + 1
display(psfs_required)

r_offsets_hdu = fits.PrimaryHDU(data=r_offsets)
r_offsets_fpath = data_dir/'psfs'/'hlc_band1_psfs_radial_samples_20230501.fits'
r_offsets_hdu.writeto(r_offsets_fpath, overwrite=True)

thetas_hdu = fits.PrimaryHDU(data=thetas.value)
thetas_fpath = data_dir/'psfs'/'hlc_band1_psfs_theta_samples_20230501.fits'
thetas_hdu.writeto(thetas_fpath, overwrite=True)


rayCGI = ray.remote(cgi.PROPERCGI)

nlam = 7
bandwidth = 0.10
minlam = wavelength_c * (1 - bandwidth/2)
maxlam = wavelength_c * (1 + bandwidth/2)
wavelengths = np.linspace( minlam, maxlam, nlam )

npol = 4
polaxis = np.array([-2, -1, 1, 2])

mode_settings = {
    'cgi_mode':'hlc',
    'use_pupil_defocus':True,    
    'use_opds':True,
}

actors = []
for i in range(nlam):
    for j in range(npol):
        actors.append(rayCGI.options(num_cpus=3).remote(wavelength=wavelengths[i],
                                                        polaxis=polaxis[j], 
                                                        **mode_settings))
        
hlc = cgi.multiCGI(actors)

hlc.set_dm1(dm1_best)
hlc.set_dm2(dm2_best)

start = time.time()

psfs_array = np.zeros( shape=( (nr-1)*nth + 1, npsf,npsf) )
print(psfs_array.shape)

count = 0
for i,r in enumerate(r_offsets): 
    for j,th in enumerate(thetas):
        xoff = r*np.cos(th)
        yoff = r*np.sin(th)

        hlc.source_offset((xoff.value,yoff.value))
        psf = hlc.snap()
#         misc.imshow1(psf, lognorm=True)
        print(count) 
        print('(r,theta) =', (r, th.value))
        print('(x,y) =', (xoff.value, yoff.value))
        print('computation time = ', time.time()-start)
        
        if r<r_offsets[1]: 
            psfs_array[0] = psf
            count += 1
            break
        else: 
            psfs_array[count] = psf
            
        count += 1

hdr = fits.Header()
hdr['PXSCLAMD'] = psf_pixelscale_lamD
hdr.comments['PXSCLAMD'] = 'pixel scale in lam0/D per pixel'
hdr['PXSCLMAS'] = psf_pixelscale_mas.value
hdr.comments['PXSCLMAS'] = 'pixel scale in mas per pixel'
hdr['PIXELSCL'] = psf_pixelscale
hdr.comments['PIXELSCL'] = 'pixel scale in meters per pixel'
hdr['CWAVELEN'] = wavelength_c.to_value(u.m)
hdr.comments['CWAVELEN'] = 'central wavelength in meters'
hdr['BANDPASS'] = bandwidth
hdr.comments['BANDPASS'] = 'bandpass as fraction of CWAVELEN'

psfs_hdu = fits.PrimaryHDU(data=psfs_array, header=hdr)

psfs_fpath = data_dir/'psfs'/'hlc_band1_psfs_20230501.fits'
psfs_hdu.writeto(psfs_fpath, overwrite=True)



