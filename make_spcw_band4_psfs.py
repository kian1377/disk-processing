import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from IPython.display import clear_output
import time
from pathlib import Path
import copy
import matplotlib.pyplot as plt

import proper
proper.prop_use_fftw(DISABLE=True)

import roman_phasec_proper
roman_phasec_proper.copy_here()

import misc
from matplotlib.patches import Circle

data_dir = Path('/groups/douglase/kians-data-files/disk-processing')

wavelength_c = 825e-9*u.m
D = 2.3631*u.m
mas_per_lamD = (wavelength_c/D*u.radian).to(u.mas)

# define desired PSF dimensions and pixelscale in units of lambda/D
npsf = 150
psf_pixelscale = 13e-6
psf_pixelscale_m = 13e-6*u.m/u.pix
psf_pixelscale_lamD = 500/825 * 1/2
psf_pixelscale_mas = psf_pixelscale_lamD*mas_per_lamD/u.pix

# psf_pixelscale_mas = 20.8*u.mas/u.pix
# psf_pixelscale_lamD = psf_pixelscale_mas.value / mas_per_lamD.value
# psf_pixelscale = 13e-6 * psf_pixelscale_lamD/(1/2)
# psf_pixelscale_m = psf_pixelscale*u.m/u.pix

disk_pixelscale_mas = 21.84*u.mas/u.pix
disk_pixelscale_lamD = psf_pixelscale_mas.value / mas_per_lamD.value
disk_pixelscale = 13e-6 * psf_pixelscale_lamD/(1/2)
disk_pixelscale_m = psf_pixelscale*u.m/u.pix

polaxis = 10

iwa = 6
owa = 20

# Create the sampling grid the PSFs will be made on
sampling1 = disk_pixelscale_lamD/2
sampling2 = disk_pixelscale_lamD/2
sampling3 = disk_pixelscale_lamD
offsets1 = np.arange(0,iwa+1,sampling1)
offsets2 = np.arange(iwa+1,owa,sampling2)
offsets3 = np.arange(owa,owa+6+sampling3,sampling3)

r_offsets = np.hstack([offsets1, offsets2, offsets3])
r_offsets_mas = r_offsets*mas_per_lamD
print(r_offsets.shape, r_offsets)

sampling_theta = 6
thetas = np.arange(0,360,sampling_theta)*u.deg
print(thetas.shape, thetas)

psfs_required = len(thetas)*len(r_offsets)
print(psfs_required)

r_offsets_hdu = fits.PrimaryHDU(data=r_offsets)
r_offsets_fpath = data_dir/'psfs'/'spcw_band4_psfs_radial_samples_20230224.fits'
r_offsets_hdu.writeto(r_offsets_fpath, overwrite=True)

thetas_hdu = fits.PrimaryHDU(data=thetas.value)
thetas_fpath = data_dir/'psfs'/'spcw_band4_psfs_theta_samples_20230224.fits'
thetas_hdu.writeto(thetas_fpath, overwrite=True)

nlam = 5
lam0 = 0.825
bandwidth = 0.1
minlam = lam0 * (1 - bandwidth/2)
maxlam = lam0 * (1 + bandwidth/2)
lam_array = np.linspace( minlam, maxlam, nlam )

dm1 = proper.prop_fits_read( roman_phasec_proper.lib_dir + r'/examples/spc_wide_band4_best_contrast_dm1.fits' )
dm2 = proper.prop_fits_read( roman_phasec_proper.lib_dir + r'/examples/spc_wide_band4_best_contrast_dm2.fits' )

options = {'cor_type':'spc-wide', # change coronagraph type to correct band
           'final_sampling_lam0':psf_pixelscale_lamD, 
           'source_x_offset':0,
           'source_y_offset':0,
           'use_fpm':1,
           'use_field_stop':1,
           'use_errors':1,
           'use_lens_errors':1,
           'use_hlc_dm_patterns':0,
           'use_dm1':1, 'dm1_m':dm1, 
           'use_dm2':1, 'dm2_m':dm2,
           'polaxis':polaxis,   
          }

psfs_array = np.zeros( shape=( (len(r_offsets)-1)*len(thetas) + 1, npsf,npsf) )

count = 0
start = time.time()
for i,r in enumerate(r_offsets): 
    opts = []
    for j,th in enumerate(thetas):
        xoff = r*np.cos(th)
        yoff = r*np.sin(th)
        options.update( {'source_x_offset':xoff.value, 'source_y_offset':yoff.value} )
    
        (wfs, pxscls_m) = proper.prop_run_multi('roman_phasec', lam_array, 256, QUIET=True, PASSVALUE=options)

        psfs = np.abs(wfs)**2
        psf = misc.pad_or_crop(np.sum(psfs, axis=0)/nlam, npsf)
        
        if r<r_offsets[1]: 
            psfs_array[0] = psf
            count += 1
            break
        else: 
            psfs_array[count] = psf
            
        print(count, time.time()-start)
        count += 1
        
hdr = fits.Header()
hdr['PXSCLAMD'] = psf_pixelscale_lamD
hdr.comments['PXSCLAMD'] = 'pixel scale in lam0/D per pixel'
hdr['PXSCLMAS'] = psf_pixelscale_mas.value
hdr.comments['PXSCLMAS'] = 'pixel scale in mas per pixel'
hdr['PIXELSCL'] = psf_pixelscale_m.value
hdr.comments['PIXELSCL'] = 'pixel scale in meters per pixel'
hdr['CWAVELEN'] = wavelength_c.value
hdr.comments['CWAVELEN'] = 'central wavelength in meters'
hdr['BANDPASS'] = bandwidth
hdr.comments['BANDPASS'] = 'bandpass as fraction of CWAVELEN'
hdr['POLAXIS'] = polaxis
hdr.comments['POLAXIS'] = 'polaxis: defined by roman_phasec_proper'

psfs_hdu = fits.PrimaryHDU(data=psfs_array, header=hdr)

psfs_fpath = data_dir/'psfs'/'spcw_band4_psfs_20230224.fits'
psfs_hdu.writeto(psfs_fpath, overwrite=True)

