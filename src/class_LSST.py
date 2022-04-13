#! /bin/python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.SimulationAPI.observation_api as observation_api
from lenstronomy.Data.imaging_data import ImageData
import lenstronomy.Util.util as util
from lenstronomy.Data.psf import PSF
from scipy.stats import skewnorm
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
import lenstronomy.Util.image_util as image_util
import pickle
plt.rc("font", family="serif")
plt.rc("text", usetex=True)

        
class LSST:

    def __init__(self, bandpass):
        """
        This class defines the telescope properties from the Vera Rubin Observatory (LSST).

        :param bandpass: LSST filter, choose from 'r', 'i' and 'z'
        """

        self.bandpass = bandpass             # choose from 'r', 'i' or 'z'
        self.exp_time = 30                   # seconds
        self.num_exposures = 1
        self.average_seeing = 0.75
        self.numPix = 48                     # cutout pixel size
        self.deltaPix = 0.2                  # pixel size in arcsec (area per pixel = deltaPix**2)
        self.psf_type = 'GAUSSIAN'           # 'GAUSSIAN', 'PIXEL', 'NONE'
        self.patch_size = self.deltaPix * self.numPix

        if bandpass == 'i':
            self.average_seeing = 0.71
            self.zero_point = 27.79
            self.limiting_magnitude = 23.9

        # Instrument properties
        self.obs_api = observation_api.SingleBand(pixel_scale=self.deltaPix, exposure_time=self.exp_time,
                                                  magnitude_zero_point=self.zero_point, read_noise=10, ccd_gain=2.3,
                                                  sky_brightness=20.48, seeing=self.average_seeing,
                                                  num_exposures=self.num_exposures, psf_type='GAUSSIAN',
                                                  kernel_point_source=None, truncation=5, data_count_unit='e-',
                                                  background_noise=None)

    def background_noise(self):
        """
        Background noise (Gaussian). Contribution of read noise (~0.5) + sky brightness (~6.1)

        :return: background noise per pixel per second (float)
        """
        sigma_bkg = self.obs_api.background_noise
        # Sky = self.obs_api._sky_brightness_cps * self.deltaPix **2 /(self.exp_time ** 0.5 * self.num_exposures ** 0.5)
        # Read = 10 / self.exp_time / (self.num_exposures ** 0.5)
        return sigma_bkg

    def grid(self, sigma_bkg):
        """
        Creates a 2D grid and provides the maximum coordinates on the grid for plotting

        :param sigma_bkg: background noise per pixel per second (float)
        :return: data_class: instance of ImageData() from Lenstronomy
                 x_grid1d, y_grid1d: 1d array of length num_pixels^2 corresponding to the x or y pixels of the image
                 min_coordinate, max_coordinate: minimum and maximum coordinates of the image array
        """
        kwargs_data = sim_util.data_configure_simple(self.numPix, self.deltaPix, self.exp_time, sigma_bkg)
        data_class = ImageData(**kwargs_data)

        max_coordinate, min_coordinate = max(data_class.pixel_coordinates[0][0]), min(data_class.pixel_coordinates[0][0])
        x_grid, y_grid = data_class.pixel_coordinates
        x_grid1d = util.image2array(x_grid)
        y_grid1d = util.image2array(y_grid)
        return data_class, x_grid1d, y_grid1d, min_coordinate, max_coordinate

    def app_mag_to_amplitude(self, app_mag_ps):
        """
        Convert apparent magnitude into amplitude parameter (brightness unit for Lenstronomy).

        :param app_mag_ps: array of length [num_images] containing the apparent magnitude for each image
        :return: array of length [num_images] containing the amplitude for each image
        """
        amp = self.obs_api.magnitude2cps(app_mag_ps) * self.exp_time

        # Filter out nans in the fluxes
        if np.any(np.isnan(amp)):
            amp[np.isnan(amp)] = 0.0

        return amp

    def get_z_theta(self, theta_min=0.1):
        """
        Download joint distribution of source redshift, lens redshift and Einstein radius for configurations where
        strong lensing occurs. Distribution is obtained from the lensed supernova simulation by Wojtek et al. (2019).

        :param theta_min: minimum accepted value of the einstein radius
        :return: z_source_list_: array containing ~ 400,000 values of the source redshift
                 z_lens_list_: array containing ~ 400,000 values of the lens redshift
                 theta_E_list_: array containing ~ 400,000 values of the einstein radius
        """

        if self.bandpass == 'i':
            zsrc_max = 1.4

        zlens_zSN_theta = np.load("../data/sample_zl_zsn_theta.npz")['zlens_zSN_theta']
        zlens_zSN_theta = np.repeat(zlens_zSN_theta, 3, axis=0)

        z_source_list_ = []
        z_lens_list_ = []
        theta_E_list_ = []

        for m in range(zlens_zSN_theta.shape[0]):
            if zlens_zSN_theta[m, 1] < zsrc_max and zlens_zSN_theta[m, 1] > zlens_zSN_theta[m, 0] and \
                    zlens_zSN_theta[m, 2] > theta_min:
                z_source_list_.append(zlens_zSN_theta[m, 1])
                z_lens_list_.append(zlens_zSN_theta[m, 0])
                theta_E_list_.append(zlens_zSN_theta[m, 2])

        return z_source_list_, z_lens_list_, theta_E_list_

    def get_inter_night_gap(self):
        """
        :return: 1D array containing the distribution of LSST inter night gaps between observations
        """
        if self.bandpass == 'i':
            with open('../data/internight_gaps_i.pickle', 'rb') as f:
                inter_night_gap = pickle.load(f)
            return inter_night_gap
        return np.nan

    def plot_inter_night_gap(self, inter_night_gap, bins=100):
        """
        Plot the distribution of LSST inter night gaps.

        :param inter_night_gap: 1D array containing the distribution of LSST inter night gaps between observations
        :param bins: number of bins for the histogram (int)
        :return: a plot displaying a histogram of the inter night gap values
        """
        plt.figure(figsize=(12, 4))
        plt.hist(inter_night_gap, bins=bins, alpha=0.6)
        plt.xlabel("Inter-night gap (days)", fontsize=22)
        plt.ylabel("Counts", fontsize=22)
        plt.title(r"Cadence for the LSST $i$-band", fontsize=25)
        plt.xlim(-10, 300)
        plt.show()

    def plot_redshifts(self, z_lens_list_, z_source_list_):
        """
        Plot the distributions of the lens and the source redshifts for the lensed supernovae.

        :param z_lens_list_: array containing the lens redshift values used in the simulation
        :param z_source_list_: array containing the source redshift values used in the simulation
        :return: a plot displaying a KDE of the source and lens redshift distributions
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sns.kdeplot(z_lens_list_, ax=ax, lw=3, color="#2e6edb", fill=True, label="Lens redshift")
        sns.kdeplot(z_source_list_, ax=ax, lw=3, color="C2", fill=True, label="Source redshift")
        ax.legend(fontsize=20)
        ax.set_xlabel(r"$z$", fontsize=25)
        ax.set_ylabel("Posterior", fontsize=25)
        fig.suptitle("Redshift distributions", fontsize=25)
        plt.show()

    def sample_z_theta(self, z_source_list_, z_lens_list_, theta_E_list_, sample, sample_index):
        """
        Selects the values of z_lens, z_source and theta_E for the current lens system.

        :param z_source_list_: array containing the source redshift values used in the simulation
        :param z_lens_list_: array containing the lens redshift values used in the simulation
        :param theta_E_list_: array containing the einstein radius values used in the simulation
        :param sample: array containing the indices of (zlens, zsrc, theta) combinations to be used in this run of
                       the simulation
        :param sample_index: counts which configuration from the sample array to select
        :return: the resulting values of z_source, z_lens and theta_E (floats) to be used for the current lens system
        """
        z_source = z_source_list_[sample[sample_index]]
        z_lens = z_lens_list_[sample[sample_index]]
        theta_E = theta_E_list_[sample[sample_index]]

        # Safety check for unlensed sources
        if z_source < 0 or z_lens < 0 or z_source < z_lens:
            return np.nan, np.nan, np.nan

        return z_source, z_lens, theta_E

    def generate_image(self, x_image, y_image, amp_ps, data_class, lens_model_class, source_model_class,
                       lens_light_model_class, kwargs_lens, kwargs_source, kwargs_lens_light, sigma_bkg):
        """
        Generate a difference image with the lensed supernova images, according to Rubin telescope settings.

        :param x_image: array of length [num_images] containing the x coordinates of the supernova images in arcsec
        :param y_image: array of length [num_images] containing the y coordinates of the supernova images in arcsec
        :param amp_ps: array of length [num_images] containing the amplitude for each image
        :param data_class: instance of ImageData() from Lenstronomy
        :param lens_model_class: Lenstronomy object returned from LensModel
        :param source_model_class: Lenstronomy object returned from LightModel corresponding to the host galaxy
        :param lens_light_model_class: Lenstronomy object returned from LightModel corresponding to the lens galaxy
        :param kwargs_lens: list of keyword arguments for the PEMD and external shear lens model
        :param kwargs_source: list of keyword arguments for the source light model
        :param kwargs_lens_light: list of keyword arguments for the lens light model
        :param sigma_bkg: background noise per pixel per second (float)
        :return: 2D array of (NumPix * DeltaPix)^2 pixels containing the image
        """
        # Sample PSF from a skewed Gaussian distribution
        fwhm = skewnorm.rvs(4.049, loc=0.552, scale=0.299)
        kwargs_psf = {'psf_type': self.psf_type, 'pixel_size': self.deltaPix, 'fwhm': fwhm}
        psf_class = PSF(**kwargs_psf)
        point_source_list = ['LENSED_POSITION']
        kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image, 'point_amp': amp_ps}]
        point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim_ = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

        # Add poisson and background noise
        poisson = image_util.add_poisson(image_sim_, exp_time=self.exp_time)
        bkg = image_util.add_background(image_sim_, sigma_bkd=sigma_bkg)
        image_sim = image_sim_ + poisson + bkg
        return image_sim


def main():

    lsst = LSST(bandpass='i')
    print("LSST limiting magnitude = ", lsst.limiting_magnitude)


if __name__ == '__main__':
    main()

