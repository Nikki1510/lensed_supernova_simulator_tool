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
from opsimsummary import SynOpSim
import pandas as pd
# plt.rc("font", family="serif")
# plt.rc("text", usetex=True)

        
class Telescope:

    def __init__(self, telescope, num_samples):
        """
        This class defines the telescope properties.

        :param telescope: choose between 'LSST' and 'ZTF'
        :param num_samples: total number of lens systems to be generated (int)
        """

        self.telescope = telescope
        self.bandpasses = ['g', 'r', 'i', 'z', 'y']

        if telescope == 'LSST':
            self.exp_time = 30                   # seconds
            self.num_exposures = 1
            self.numPix = 48                     # cutout pixel size
            self.deltaPix = 0.2                  # pixel size in arcsec (area per pixel = deltaPix**2)
            self.psf_type = 'GAUSSIAN'           # 'GAUSSIAN', 'PIXEL', 'NONE'
            self.patch_size = self.deltaPix * self.numPix

        # Create telescope sky pointings
        ra_pointings, dec_pointings = self.create_sky_pointings(N=num_samples)

        # Initialise OpSim Summary (SynOpSim) generator
        print("Setting up OpSim Summary generator...")
        self.gen = self.initialise_opsim_summary(ra_pointings, dec_pointings)

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
                      'limiting_magnitude': 25.0,
                      'limiting_magnitude_wojtak': 24.5}

            LSST_r = {'magnitude_zero_point': 28.13,
                      'average_seeing': 0.73,
                      'sky_brightness': 21.2,
                      'limiting_magnitude': 24.7,
                      'limiting_magnitude_wojtak': 24.2}

            LSST_i = {'magnitude_zero_point': 27.79,
                      'average_seeing': 0.71,
                      'sky_brightness': 20.48,
                      'limiting_magnitude': 24.0,
                      'limiting_magnitude_wojtak': 23.6}

            LSST_z = {'magnitude_zero_point': 27.40,
                      'average_seeing': 0.69,
                      'sky_brightness': 19.6,
                      'limiting_magnitude': 23.3,
                      'limiting_magnitude_wojtak': 22.8}

            LSST_y = {'magnitude_zero_point': 26.58,
                      'average_seeing': 0.68,
                      'sky_brightness': 18.61,
                      'limiting_magnitude': 22.1,
                      'limiting_magnitude_wojtak': 22.0}

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
        limiting_magnitude_wojtak = obs_dict['limiting_magnitude_wojtak']
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

        return obs_api, limiting_magnitude, sigma_bkg, zero_point, limiting_magnitude_wojtak

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

    def create_sky_pointings(self, N, dec_low=-90, dec_high=40):
        """
        Creates random points on a sphere (limited between dec_low and dec_high).
        Acception fraction of points is around 2/3, so sample ~1.6 times as many points.

        :param N: number of desired points inside the LSST footprint. The actual initiated number is 1.6 times higher
        :param dec_low: lower declination limit
        :param dec_high: upper declination limit
        :return: two arrays containing the x-coordinates (right ascension) and y-coordinates (declination) of random sky pointings
        """

        if N < 10:
            sample_number = int(N * 5)
        else:
            sample_number = int(N * 1.6)

        ra_points = np.random.uniform(low=0, high=360, size=sample_number)
        dec_points = np.arcsin(2 * np.random.uniform(size=sample_number) - 1) / np.pi * 180

        dec_selection = (dec_points > dec_low) & (dec_points < dec_high)
        ra_points = ra_points[dec_selection]
        dec_points = dec_points[dec_selection]

        return ra_points, dec_points

    def initialise_opsim_summary(self, ra_pointings, dec_pointings):
        """
        Itialise the generator for OpSim Summary SynOpSim. This will allow to draw random cadence realisations.
        :param ra_pointings: array with the x-coordinates (right ascension) of random sky pointings
        :param dec_pointings: array with the y-coordinates (declination) of random sky pointings
        :return: OpSim Summary generator for a given OpSim database and sky pointings
        """
        # Location of the OpSim database used for the cadence realisations
        myopsimv3 = '../data/OpSim_databases/baseline_v3.0_10yrs.db'

        synopsim = SynOpSim.fromOpSimDB(myopsimv3, opsimversion='fbsv2', usePointingTree=True, use_proposal_table=False,
                                  subset='unique_all')

        gen = synopsim.pointingsEnclosing(ra_pointings, dec_pointings, circRadius=0., pointingRadius=1.75,
                                          usePointingTree=True)
        return gen

    def opsim_observation(self, gen):
        """
        Function to draw one random cadence realisation for 1 sky position from the OpSim database.
        If a sky pointing is outside of the LSST footprint, the function continues until the point is in the footprint.
        :param gen: OpSim Summary generator for a given OpSim database and sky pointings
        :return: 2 floats with the right ascension and declination of the observation, and 5 arrays containing the
        observation times (in MJD), filters, PSF FWHM ('seeingFwhmGeom'), limiting magnitude, and sky brightness for
        10 years of LSST observations.
        """

        while True:

            obs = next(gen)
            opsim_ra = np.mean(obs['fieldRA'])
            opsim_dec = np.mean(obs['fieldDec'])
            opsim_times = np.array(obs['expMJD'])

            if np.isnan(opsim_ra) or np.isnan(opsim_dec):
                continue

            if len(opsim_times) == 0:
                continue

            obs = obs.sort_values(by=['expMJD'])

            opsim_times = np.array(obs['expMJD'])
            opsim_filters = np.array(obs['filter'])
            opsim_psf = np.array(obs['seeingFwhmGeom'])
            opsim_lim_mag = np.array(obs['fiveSigmaDepth'])
            opsim_sky_brightness = np.array(obs['filtSkyBrightness'])
            break

        return opsim_ra, opsim_dec, opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness

    def select_observation_time_period(self, times, filters, psf, lim_mag, sky_brightness, mjd_low, mjd_high=61325):
        """
        Function to limit the LSST observations to a shorter time duration. The default selection excludes everything
        after the third year of observations (MJD = 61325).
        :param times: array with observation times (in MJD)
        :param filters: array with filters/bandpasses
        :param psf: array with PSF sizes
        :param lim_mag: array with limiting magnitudes
        :param sky_brightness: array with sky brightnesses
        :param mjd_low: lower threshold (everything before this date will be discarded)
        :param mjd_high: upper threshold (everything after this date will be discarded). Default: end of year 3
        :return: the same arrays, limited to dates between mjd_low and mjd_high
        """

        indices = (times > mjd_low) & (times < mjd_high)

        return times[indices], filters[indices], psf[indices], lim_mag[indices], sky_brightness[indices]

    def coadds(self, opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness):
        """
        Calculates nightly coadds if observations are taken on the same day in the same filter.
        :param opsim_times: array with observation times (days relative to the SN peak)
        :param opsim_filters: array with filters/bandpasses corresponding to opsim_times
        :param opsim_psf: array with PSF sizes
        :param opsim_lim_mag: array with limiting magnitudes
        :param opsim_sky_brightness: array with sky brightnesses
        :return: arrays with observation times, filters, psf, coadded limiting magnitudes, sky brightness.
        """

        coadd_times, coadd_filters, coadd_psf, coadd_lim_mag, coadd_sky_brightness = [], [], [], [], []
        IDs = np.arange(0, len(opsim_times))

        N_coadds = []
        ID_list = []

        for t1 in range(len(opsim_times)):

            # Check if observation was already coadded
            if IDs[t1] in ID_list:
                continue

            # Add limiting magnitude, time, and ID
            lim_mag_list = [opsim_lim_mag[t1]]
            times_list = [opsim_times[t1]]
            ID_list.append(IDs[t1])

            for t2 in range(t1+1, len(opsim_times)):

                # Same day?
                if opsim_times[t2] - opsim_times[t1] >= 1:
                    break

                # Same filter?
                if opsim_filters[t1] == opsim_filters[t2]:
                    lim_mag_list.append(opsim_lim_mag[t2])
                    times_list.append(opsim_times[t2])
                    ID_list.append(IDs[t2])

            # Perform coadds
            coadd_lim_mag.append(self.calculate_coadd(lim_mag_list))
            coadd_times.append(np.mean(times_list))
            N_coadds.append(len(times_list))
            coadd_filters.append(opsim_filters[t1])
            coadd_psf.append(opsim_psf[t1])
            coadd_sky_brightness.append(opsim_sky_brightness[t1])

        coadd_lim_mag = np.array(coadd_lim_mag)
        coadd_times = np.array(coadd_times)
        coadd_filters = np.array(coadd_filters)
        coadd_psf = np.array(coadd_psf)
        coadd_sky_brightness = np.array(coadd_sky_brightness)
        N_coadds = np.array(N_coadds)

        return coadd_times, coadd_filters, coadd_psf, coadd_lim_mag, coadd_sky_brightness, N_coadds

    def calculate_coadd(self, lim_mag_list):
        """
        Calculates the new limiting magnitude by combining the limiting magnitudes in lim_mag_list.
        Formula from https://smtn-016.lsst.io

        :param lim_mag_list: list containing the limiting magnitudes that need to be coadded
        :return: the coadded limiting magnitude (float)
        """

        lim_mag_array = np.array(lim_mag_list)
        lim_mag_new = 1.25 * np.log10(np.sum(10**(0.8 * lim_mag_array)))
        return lim_mag_new

    def determine_survey(self, Nobs_3yr, Nobs_10yr, obs_start):
        """
        Determine whether the observation belongs to the WFD, DDF or galactic plane and pole region.
        If WFD, determine whether the cadence is rolling and the coordinates belong to the active or background region.

        :param Nobs_3yr: Number of visits after 3 years of LSST observations
        :param Nobs_10yr: Number of visits after 10 years of LSST observations
        :param obs_start: MJD of the start of the observation
        :return:
        """

        if Nobs_10yr < 400:
            survey = 'galactic plane'
            rolling = np.nan
        elif Nobs_10yr > 1000:
            survey = 'DDF'
            rolling = np.nan
        else:
            survey = 'WFD'

            if obs_start <= 60768:
                rolling = 'no rolling'

            else:
                if Nobs_3yr > 255:
                    rolling = 'active'
                else:
                    rolling = 'background'

        return survey, rolling

    def get_weather(self, zeropoint, skysig, psf_sig):
        """
        Calculates the flux from the sky signal and the limiting magnitude (5 sigma depth)

        :param zeropoint: instrument zeropoint in magnitudes (weather dependent)
        :param skysig: signal from the sky in electron counts/pixel (weather dependent)
        :param psf_sig: sigma of a Gaussian fit to the PSF (weather dependent)
        :return: flux_skysig in electron counts, limiting magnitude
        """
        # Calculate effective PSF area
        psf_area = 4 * np.pi * psf_sig ** 2  # pixel area
        flux_skysig = skysig * psf_area ** 0.5  # e- counts
        lim_mag = zeropoint - 2.5 * np.log10(5 * flux_skysig)

        return flux_skysig, lim_mag

    def plot_redshifts(self):
        """
        Plot the distributions of the lens and the source redshifts for the lensed supernovae.

        :return: a plot displaying a KDE of the source and lens redshift distributions
        """

        z_source_list_, z_lens_list_, theta_E_list_ = self.load_z_theta(theta_min=0.05)

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
        bandpasses = ['r', 'i', 'z', 'y']

        for band in range(4):
            seeing_params = self.get_seeing_params(bandpasses[band])
            ax.plot(x_range, stats.skewnorm.pdf(x_range, seeing_params['s'], seeing_params['mean'], seeing_params['sigma']),
                    color=colours[band], lw=3, label=r"$%s$ -band, median = %.2f''" % (bandpasses[band],
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
                       lens_light_model_class, kwargs_lens, kwargs_source, kwargs_lens_light, band, psf, Noise=True):
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
        :param psf: FWHM of the PSF for this observation
        :param Noise: Bool. if True: add noise to the image
        :return: 2D array of (NumPix * DeltaPix)^2 pixels containing the image
        """

        _, limiting_mag, sigma_bkg, _, _ = self.single_band_properties(band)
        data_class, x_grid1d, y_grid1d, min_coordinate, max_coordinate = self.grid(sigma_bkg)

        # seeing_params = self.get_seeing_params(band)
        # fwhm = skewnorm.rvs(seeing_params['s'], seeing_params['mean'], seeing_params['sigma'])

        kwargs_psf = {'psf_type': self.psf_type, 'pixel_size': self.deltaPix, 'fwhm': psf}
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

