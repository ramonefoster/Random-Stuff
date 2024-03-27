import numpy as np
from astropy.io import fits
from photutils.detection import DAOStarFinder
from photutils.profiles import RadialProfile

import matplotlib.pyplot as plt

from astropy.modeling.models import Gaussian2D
from astropy.visualization import simple_norm

hdul = fits.open('hr7134_0_Clear.fits')
image = hdul[0].data

img_shape = image.shape
if img_shape[0] == 1: image = image[0]
threshold = 3  # Detection threshold (in standard deviations)
fwhm = 8  # Expected full width at half maximum (FWHM) of stars (in pixels)

# Find stars using DAOStarFinder
daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, brightest=10)
sources = daofind(image)
stars = list(zip(sources['xcentroid'], sources['ycentroid'])) 

edge_radii = np.arange(26)
rp = RadialProfile(image, stars[7], edge_radii, mask=None)
rp.gaussian_fit 

print(rp.gaussian_fwhm)
norm = simple_norm(image, 'sqrt')
plt.figure(figsize=(5, 5))
plt.imshow(image, norm=norm)

rp.apertures[5].plot(color='C0', lw=2)
rp.apertures[10].plot(color='C1', lw=2)

plt.show()