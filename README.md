# disk-processing
Repo for the disk processing pipeline to simulate images of exozodiacal disks. 

- create_offset_psf_array_with_proper.ipynb
  - This file is for creating the array of off-axis PSFs with the PROPER Fresnel models of the Roman CGI. 
  - Options for the PSFs can be changed such as the polarization scenario, the pixelscale, the DM settings, and the use of OPDs.

- create_interpolated_psfs_matrix.ipynb
  - This file is where the array of model based PSFs is rotated and interpolated to estimate a PSF for each pixel position of the pixels in the disk models.
  - The amount of interpolated PSFs used is based on the dimensions of the disk model.
  - The dimensions of the interpolated PSF matrix is (npsf^2 x ndisk^2)

- run_disk_simulation.ipynb
  - This file is where the disk and interpolated PSF matrix are loaded in to run the simulation.
  - The simulation is just a vector-matrix multiplication between the I-PSF matrix and the flattened array of the disk model (a dot-product is used in the code).

To create your own set of PSFs, you need to install roman_phasec_proper, available at https://sourceforge.net/projects/cgisim/files/. 

