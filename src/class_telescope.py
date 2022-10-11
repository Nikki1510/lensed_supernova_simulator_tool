#! /bin/python3
from functions import _clean_obj_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
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
import scipy.stats as stats
import pandas as pd
plt.rc("font", family="serif")
plt.rc("text", usetex=True)

        
class Telescope:

    def __init__(self, telescope, bandpasses):
        """
        This class defines the telescope properties.

        :param bandpasses: list containing bandpasses that will be used, choose from 'g', 'r', 'i', 'z' and 'y'
        :param telescope: choose between 'LSST' and 'ZTF'
        """

        self.telescope = telescope
        self.bandpasses = bandpasses             # choose from 'g', 'r', 'i', 'z' and 'y'

        if telescope == 'LSST':
            self.exp_time = 30                   # seconds
            self.num_exposures = 1
            self.numPix = 48                     # cutout pixel size
            self.deltaPix = 0.2                  # pixel size in arcsec (area per pixel = deltaPix**2)
            self.psf_type = 'GAUSSIAN'           # 'GAUSSIAN', 'PIXEL', 'NONE'
            self.patch_size = self.deltaPix * self.numPix

        # Create: elif telescope == 'ZTF'

    def single_band_properties(self, band):
        """
        Returns instrument properties for a specific bandpass.
        Limiting magnitudes from https://www.lsst.org/scientists/keynumbers

        :param band: chosen bandpass to compute the instrument properties
        :return: obs_api, limiting magnitude and background noise
        """

        if self.telescope == 'LSST':

            LSST_g = {'magnitude_zero_point': 28.30,
                      'average_seeing': 0.77,
                      'sky_brightness': 22.26,
                      'limiting_magnitude': 25.0}

            LSST_r = {'magnitude_zero_point': 28.13,
                      'average_seeing': 0.73,
                      'sky_brightness': 21.2,
                      'limiting_magnitude': 24.7}

            LSST_i = {'magnitude_zero_point': 27.79,
                      'average_seeing': 0.71,
                      'sky_brightness': 20.48,
                      'limiting_magnitude': 24.0}

            LSST_z = {'magnitude_zero_point': 27.40,
                      'average_seeing': 0.69,
                      'sky_brightness': 19.6,
                      'limiting_magnitude': 23.3}

            LSST_y = {'magnitude_zero_point': 26.58,
                      'average_seeing': 0.68,
                      'sky_brightness': 18.61,
                      'limiting_magnitude': 22.1}

            if band == 'g':
                obs_dict = LSST_g
            elif band == 'r':
                obs_dict = LSST_r
            elif band == 'i':
                obs_dict = LSST_i
            elif band == 'z':
                obs_dict = LSST_z
            elif band == 'y':
                obs_dict = LSST_y
            else:
                raise ValueError("band %s not supported! Choose 'g', 'r', 'i', 'z' or 'y' for LSST." % band)

        # Create elif self.telescope == 'ZTF'

        zero_point = obs_dict['magnitude_zero_point']
        average_seeing = obs_dict['average_seeing']
        limiting_magnitude = obs_dict['limiting_magnitude']
        sky_brightness = obs_dict['sky_brightness']

        # Instrument properties
        obs_api = observation_api.SingleBand(pixel_scale=self.deltaPix, exposure_time=self.exp_time,
                                             magnitude_zero_point=zero_point, read_noise=10, ccd_gain=2.3,
                                             sky_brightness=sky_brightness, seeing=average_seeing,
                                             num_exposures=self.num_exposures, psf_type='GAUSSIAN',
                                             kernel_point_source=None, truncation=5, data_count_unit='e-',
                                             background_noise=None)

        # Background noise per pixel per second
        # Noise is Gaussian and consists of read noise (~0.5 for LSST i-band) + sky brightness (~6.1).
        # Sky = self.obs_api._sky_brightness_cps * self.deltaPix **2 /(self.exp_time ** 0.5 * self.num_exposures ** 0.5)
        # Read = 10 / self.exp_time / (self.num_exposures ** 0.5)
        sigma_bkg = obs_api.background_noise

        return obs_api, limiting_magnitude, sigma_bkg

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

    def app_mag_to_amplitude(self, app_mag_ps, band):
        """
        Convert apparent magnitude into amplitude parameter (brightness unit for Lenstronomy).

        :param app_mag_ps: array of length [num_images] containing the apparent magnitude for each image
        :param band: chosen bandpass to convert apparent magnitude to amplitude
        :return: array of length [num_images] containing the amplitude for each image
        """
        obs_api = self.single_band_properties(band)[0]
        amp = obs_api.magnitude2cps(app_mag_ps) * self.exp_time

        # Filter out nans in the fluxes
        if np.any(np.isnan(amp)):
            amp[np.isnan(amp)] = 0.0

        return amp

    def load_z_theta(self, theta_min=0.1, zsrc_max=1.5):
        """
        Download joint distribution of source redshift, lens redshift and Einstein radius for configurations where
        strong lensing occurs. Distribution is obtained from the lensed supernova simulation by Wojtek et al. (2019).

        :param theta_min: minimum accepted value of the einstein radius
        :param zsrc_max: maximum accepted value of the source redshift
        :return: z_source_list_: array containing ~ 400,000 values of the source redshift
                 z_lens_list_: array containing ~ 400,000 values of the lens redshift
                 theta_E_list_: array containing ~ 400,000 values of the einstein radius
        """

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

    def load_cadence(self, small_sample=False):
        """
        Cadence simulations by Catarina Alves for supernovae in LSST.

        :param small_sample: bool. if True: use a smaller sample of 64 observation sequences
                                   if False: use full set of 78159 observation sequences
        :return: 1D array containing the distribution of LSST inter night gaps between observations.
        """
        if small_sample:
            with open('../data/catarina_cadence/file_train_wfd_nikki_000.pckl', 'rb') as f:
                cadence = pickle.load(f)
        else:
            with open('../data/catarina_cadence/file_test_wfd_nikki_000.pckl', 'rb') as f:
                cadence = pickle.load(f)

        obs_times = []
        obs_filters = []

        tqdm._instances.clear()
        pbar = tqdm(total=len(cadence))

        for c in range(len(cadence)):
            cadence_clean = _clean_obj_data(cadence[c])
            mask = np.nonzero(np.in1d(cadence_clean['filter'], self.bandpasses))[0]
            obs_times_temp = cadence_clean[mask]['mjd']
            obs_filters_temp = cadence_clean[mask]['filter']
            obs_times.append(obs_times_temp)
            obs_filters.append(obs_filters_temp)

            pbar.update(1)

        return obs_times, obs_filters

    def get_total_obs_times(self, obs_times, obs_filters):
        """
        Compute lists with the observation times of all objects together.

        :param obs_times: contains for each observation the observation times
        :param obs_filters: contains for each observation the filters/bandpasses used
        :return: lists with all observation times and the ones in the r, i, z, and y bands
        """
        obs_all = []
        obs_r = []
        obs_i = []
        obs_z = []
        obs_y = []

        for lens in range(len(obs_times)):
            obs_all += list(obs_times[lens])

            for obs in range(len(obs_times[lens])):

                if obs_filters[lens][obs] == 'r':
                    obs_r.append(obs_times[lens][obs])

                if obs_filters[lens][obs] == 'i':
                    obs_i.append(obs_times[lens][obs])

                if obs_filters[lens][obs] == 'z':
                    obs_z.append(obs_times[lens][obs])

                if obs_filters[lens][obs] == 'y':
                    obs_y.append(obs_times[lens][obs])

        return obs_all, obs_r, obs_i, obs_z, obs_y

    def plot_cadence(self, obs_times, obs_all, obs_r, obs_i, obs_z, obs_y, bins=100):
        """
        Plot the distribution of LSST inter night gaps and return the minimum and maximum observation dates.

        :param obs_times: contains for each observation the observation times
        :param obs_all: a list with the observation times of all objects together
        :param obs_r: a list with the observation times in the r-band of all objects together
        :param obs_i: a list with the observation times in the i-band of all objects together
        :param obs_z: a list with the observation times in the z-band of all objects together
        :param obs_y: a list with the observation times in the y-band of all objects together
        :param bins: the number of bins of the histograms
        :return: a plot displaying a histogram of the inter night gap values
        """

        plt.figure(figsize=(12, 4))
        plt.hist(obs_all, bins=bins, color='black', alpha=0.15, label=r"all")
        plt.hist(obs_r, bins=bins, color='#4daf4a', alpha=0.3, label=r"$r$-band")
        plt.hist(obs_i, bins=bins, color='#e3c530', alpha=0.3, label=r"$i$-band")
        plt.hist(obs_z, bins=bins, color='#ff7f00', alpha=0.3, label=r"$z$-band")
        plt.hist(obs_y, bins=bins, color='#e41a1c', alpha=0.3, label=r"$y$-band")
        plt.xlabel("MJD", fontsize=20)
        plt.ylabel("Observations", fontsize=20)
        plt.title(r"Baseline v2.0 cadence for " + str(len(obs_times)) + " objects", fontsize=25)
        plt.legend(loc=(1.04, 0.36), fontsize=18)
        # plt.savefig("../results/figures/Cadence_78159_objects.pdf", transparent=True, bbox_inches='tight')
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
        ax.set_ylabel("Posterior", fontsize=20)
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

    def get_seeing_params(self, band):
        """
        Return the shape parameter, mean and sigma of a skewed Gaussian that is fitted to the seeing distribution.

        :param band: the observing band in which the seeing distribution is approximated
        :return: a dictionary containing the shape parameter (s), mean and sigma of the skewed Gaussian distribution
        """

        if self.telescope == 'LSST':
            # if band == 'g':
            #    seeing = {'s': 5.2626,
            #              'mean': 0.6795,
            #              'sigma': 0.4241}

            if band == 'r':
                seeing = {'s': 5.6438,
                          'mean': 0.6492,
                          'sigma': 0.3923}
            elif band == 'i':
                seeing = {'s': 5.9080,
                          'mean': 0.6321,
                          'sigma': 0.3725}
            elif band == 'z':
                seeing = {'s': 6.2456,
                          'mean': 0.6175,
                          'sigma': 0.3587}
            elif band == 'y':
                seeing = {'s': 5.5672,
                          'mean': 0.6111,
                          'sigma': 0.3386}

        return seeing

    def plot_seeing_distributions(self):
        """
        Plots the seeing distributions in all bands.
        :return: figure displaying the seeing distributions
        """

        colours = ['#4daf4a', '#e3c530', '#ff7f00', '#e41a1c']
        x_range = np.linspace(0, 5, 1000)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        for band in range(len(self.bandpasses)):
            seeing_params = self.get_seeing_params(self.bandpasses[band])
            ax.plot(x_range, stats.skewnorm.pdf(x_range, seeing_params['s'], seeing_params['mean'], seeing_params['sigma']),
                    color=colours[band], lw=3, label=r"$%s$ -band, median = %.2f''" % (self.bandpasses[band],
                    stats.skewnorm.median(seeing_params['s'], seeing_params['mean'], seeing_params['sigma'])))

        # ax.legend(loc=(1.04, 0.45), fontsize=18)
        ax.legend(loc='upper right', fontsize=15)
        ax.set_xlabel("PSF fwhm (arcsec)", fontsize=20)
        ax.set_ylabel("Posterior", fontsize=20)
        ax.set_xlim(0.2, 2)
        fig.suptitle("Seeing distributions", fontsize=25)
        # plt.savefig("../results/figures/Seeing_distributions_rizy.pdf", transparent=True, bbox_inches='tight')
        plt.show()

    def generate_image(self, x_image, y_image, amp_ps, lens_model_class, source_model_class,
                       lens_light_model_class, kwargs_lens, kwargs_source, kwargs_lens_light, band, Noise=True):
        """
        Generate a difference image with the lensed supernova images, according to Rubin telescope settings.

        :param x_image: array of length [num_images] containing the x coordinates of the supernova images in arcsec
        :param y_image: array of length [num_images] containing the y coordinates of the supernova images in arcsec
        :param amp_ps: array of length [num_images] containing the amplitude for each image
        :param lens_model_class: Lenstronomy object returned from LensModel
        :param source_model_class: Lenstronomy object returned from LightModel corresponding to the host galaxy
        :param lens_light_model_class: Lenstronomy object returned from LightModel corresponding to the lens galaxy
        :param kwargs_lens: list of keyword arguments for the PEMD and external shear lens model
        :param kwargs_source: list of keyword arguments for the source light model
        :param kwargs_lens_light: list of keyword arguments for the lens light model
        :param band: bandpass at which to generate the image. choose from 'g', 'r', 'i', 'z', 'y' for LSST
        :return: 2D array of (NumPix * DeltaPix)^2 pixels containing the image
        """

        _, limiting_mag, sigma_bkg = self.single_band_properties(band)
        data_class, x_grid1d, y_grid1d, min_coordinate, max_coordinate = self.grid(sigma_bkg)

        seeing_params = self.get_seeing_params(band)

        # Sample PSF from a skewed Gaussian distribution
        fwhm = skewnorm.rvs(seeing_params['s'], seeing_params['mean'], seeing_params['sigma'])
        kwargs_psf = {'psf_type': self.psf_type, 'pixel_size': self.deltaPix, 'fwhm': fwhm}
        psf_class = PSF(**kwargs_psf)
        point_source_list = ['LENSED_POSITION']
        kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image, 'point_amp': amp_ps}]
        point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])
        kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

        imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class, lens_light_model_class,
                                point_source_class, kwargs_numerics=kwargs_numerics)
        image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)

        # Add poisson and background noise
        if Noise:
            poisson = image_util.add_poisson(image_sim, exp_time=self.exp_time)
            bkg = image_util.add_background(image_sim, sigma_bkd=sigma_bkg)
            image_sim = image_sim + poisson + bkg

        return image_sim


def main():

    lsst = Telescope(telescope='LSST', bandpasses=['i'])
    print("LSST exp time = ", lsst.exp_time)


if __name__ == '__main__':
    main()

