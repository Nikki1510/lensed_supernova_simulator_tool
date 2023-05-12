#! /bin/python3
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
from scipy.sparse import csr_matrix
from functions import create_dataframe, write_to_df, get_time_delay_distance
from class_lens import Lens
from class_supernova import Supernova
from class_microlensing import Microlensing
from class_visualisation import Visualisation
from class_telescope import Telescope
from class_timer import Timer


class Simulations:

    # def __init__(self):

    def initialise_parameters(self, lsst, z_source_list_, z_lens_list_, theta_E_list_, sample, sample_index, fixed_H0):
        """
        Initialise supernova and cosmology parameters.

        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param z_source_list_: array containing ~ 400,000 values of the source redshift
        :param z_lens_list_: array containing ~ 400,000 values of the lens redshift
        :param theta_E_list_: array containing ~ 400,000 values of the Einsten radius
        :param sample: array containing the indices of (zlens, zsrc, theta) combinations to be used in this run of
                       the simulation
        :param sample_index: counts which configuration from the sample array to select
        :param fixed_H0: bool. if True: H0 is kept to a fixed value (evaluationsest). if False: H0 varies (training/test set)
        :return: values for z_source, z_lens, theta_E, H_0, cosmo, time_delay_distance, source_x, source_y used in this run
        """

        # Sample lens configuration and cosmology
        z_source, z_lens, theta_E = lsst.sample_z_theta(z_source_list_, z_lens_list_, theta_E_list_,
                                                        sample, sample_index)

        if fixed_H0:
            H_0 = 67.8  # Planck 2018 cosmology
        else:
            H_0 = np.random.uniform(20.0, 100.0)

        cosmo = FlatLambdaCDM(H0=H_0, Om0=0.315)
        time_delay_distance = get_time_delay_distance(z_source, z_lens, cosmo)
        source_x = np.random.uniform(-theta_E, theta_E)
        source_y = np.random.uniform(-theta_E, theta_E)

        return z_source, z_lens, theta_E, H_0, cosmo, time_delay_distance, source_x, source_y

    def check_mult_method_peak(self, supernova, sep, lsst, model, macro_mag):
        """
        Check whether multiple images of the lensed SN are visible (evaluated at peak brightness for each image).

        :param supernova: class that contains supernova functions
        :param sep: maximum separation between the SN images
        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param model: SNcosmo model for the supernova light curve
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :return: mult_method_peak: bool. if True: lensed SN passes the image multiplicity method
        """

        mult_method_peak = False

        # Is maximum image separation between 0.5 and 4.0 arcsec?
        if sep > 0.5 and sep < 4.0:

            # Check peak brightness and flux ratio: detectable?
            if supernova.check_detectability_peak(lsst, model, macro_mag, 0.0, False):
                mult_method_peak = True

        return mult_method_peak

    def check_mag_method_peak(self, td_images, lsst, supernova, model, macro_mag, add_microlensing, cosmo, z_lens, M_i,
                              mag_gap):
        """
        Check whether the unresolved supernova images are brighter than a typical type Ia SN at the lens redshift.

        :param td_images: array of length [num_images] containing the relative time delays between the supernova images
        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param supernova: class that contains supernova functions
        :param model: SNcosmo model for the supernova light curve
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param add_microlensing: bool. if False: no microlensing. if True: also add microlensing to the peak
        :param cosmo: instance of astropy containing the background cosmology
        :param z_lens: redshift of the lens galaxy (float)
        :param M_i: absolute magnitude of the lensed SN in the i-band
        :param mag_gap: number of magnitudes the lensed SN must be brighter than a typical Ia at the lens redshift
        :return: mag_method_peak: bool. if True: lensed SN passes the magnification method
        """

        mag_method_peak = False

        # Sample different times to get the brightest combined flux
        time_range = np.linspace(min(td_images), max(td_images), 100)
        app_mag_i_unresolved = 50
        for t in time_range:
            lim_mag_i = lsst.single_band_properties('i')[1]
            app_mag_i_temp = supernova.get_app_magnitude(model, t, macro_mag, td_images, np.nan, lsst,
                                                                  'i', lim_mag_i, add_microlensing=False)[0]
            app_mag_i_unresolved_temp = supernova.get_mags_unresolved(app_mag_i_temp, lsst, ['i'], 24.0, filler=None)[0]

            if app_mag_i_unresolved_temp < app_mag_i_unresolved:
                app_mag_i_unresolved = app_mag_i_unresolved_temp

        # Checks whether eq. 1 from Wojtak et al. (2019) holds
        if app_mag_i_unresolved < M_i + cosmo.distmod(z_lens).value + mag_gap:
            mag_method_peak = True

        return mag_method_peak

    def check_mult_method(self, supernova, sep, lsst, model, macro_mag, obs_mag, obs_lim_mag, obs_filters, micro_peak,
                          add_microlensing):
        """
        Check whether multiple images of the lensed SN are visible.

        :param supernova: class that contains supernova functions
        :param sep: maximum separation between the SN images
        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param model: SNcosmo model for the supernova light curve
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param obs_mag: array of shape [N_observations, N_images] that contains the apparent magnitudes for each
            observation and each image (perterbed by weather)
        :param obs_lim_mag: array of length N_observations containing the limiting magnitude (5 sigma depth)
        :param obs_filters: array of length N_observations containing the bandpasses for each observation
        :param micro_peak: array of length [num_images] containing the microlensing contributions at light curve peak
        :param add_microlensing: bool. if False: no microlensing. if True: also add microlensing to the peak
        :return: mult_method: bool. if True: lensed SN passes the image multiplicity method
        """

        mult_method = False

        if sep > 0.5 and sep < 4.0:
            # Check maximum brightness and flux ratio: detectable?
            if supernova.check_detectability(lsst, model, macro_mag, obs_mag, obs_lim_mag, obs_filters, micro_peak,
                                             add_microlensing):
                mult_method = True

        return mult_method

    def check_mag_method(self, supernova, app_mag_i_model, lsst, M_i, cosmo, z_lens, mag_gap):
        """
        Check whether the unresolved supernova images are brighter than a typical type Ia SN at the lens redshift.

        :param supernova: class that contains supernova functions
        :param app_mag_i_model: array of shape [N_observations, N_images] that contains the model apparent magnitudes
               of the lensed SN in the i-band (without taking into account weather)
        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param M_i: absolute magnitude of the lensed SN in the i-band
        :param cosmo: instance of astropy containing the background cosmology
        :param z_lens: redshift of the lens galaxy (float)
        :param mag_gap: number of magnitudes the lensed SN must be brighter than a typical Ia at the lens redshift
        :return: mag_method: bool. if True: lensed SN passes the magnification method
        """

        mag_method = False

        app_mag_i_obs_min = np.min(supernova.get_mags_unresolved(app_mag_i_model, lsst,
                                                                 ['i' for i in range(len(app_mag_i_model))],
                                                                 [24 for i in range(len(app_mag_i_model))],
                                                                 filler=np.nan)[0])

        if app_mag_i_obs_min < M_i + cosmo.distmod(z_lens).value + mag_gap:
            mag_method = True

        return mag_method

    def get_observations(self, lsst, supernova, gen, model, td_images, x_image, y_image, z_source, macro_mag,
                         lens_model_class, source_model_class, lens_light_model_class, kwargs_lens, kwargs_source,
                         kwargs_lens_light, add_microlensing, microlensing, micro_contributions, obs_upper_limit,
                         Show, N_tries=20):
        """

        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param supernova: class that contains supernova functions
        :param gen: OpSim Summary generator for a given OpSim database and sky pointings
        :param model: SNcosmo model for the supernova light curve
        :param td_images: array of length [num_images] containing the relative time delays between the supernova images
        :param x_image: array of length [num_images] containing the x coordinates of the supernova images in arcsec
        :param y_image: array of length [num_images] containing the y coordinates of the supernova images in arcsec
        :param z_source: redshift of the host galaxy (float)
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param lens_model_class: Lenstronomy object returned from LensModel
        :param source_model_class: Lenstronomy object returned from LightModel corresponding to the host galaxy
        :param lens_light_model_class: Lenstronomy object returned from LightModel corresponding to the lens galaxy
        :param kwargs_lens: list of keyword arguments for the PEMD and external shear lens model
        :param kwargs_source: list of keywords arguments for the host light model
        :param kwargs_lens_light: list of keywords arguments for the lens light model
        :param add_microlensing: bool. if False: no microlensing. if True: also add microlensing to the peak
        :param microlensing: class that contains functions related to microlensing
        :param micro_contributions: list of length [num_images] containing microlensing dictionaries
        :param obs_upper_limit: minimum number of observations (below which systems are discarded)
        :param Show: bool. if True: figures and print statements show the properties of the lensed SN systems
        :param N_tries: Number of times different cadence realisations should be tried for this lensed SN system
        :return: obs_days, obs_filters, obs_skybrightness, obs_lim_mag, obs_psf, obs_snr, obs_N_coadds, model_mag,
                 obs_mag, app_mag_i_model, obs_mag_error, obs_start, time_series, coords
        """

        for cadence_try in range(N_tries):

            ra, dec, opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness = lsst.opsim_observation(
                gen)

            Nobs_10yr = len(opsim_times)
            Nobs_3yr = len(opsim_times[opsim_times < 61325])

            coords = np.array([ra, dec])

            # Start and end time of the lensed supernova
            start_sn = model.mintime() + min(td_images)
            end_sn = model.maxtime() + max(td_images)

            # Start the SN randomly between the start of 1st season and end of 3rd one (dates from Catarina)
            offset = np.random.randint(60220, 61325 - 50)

            # """
            if Show:
                plt.figure(5)
                plt.hist(opsim_times, bins=100)
                plt.axvline(x=offset - start_sn, color='C3')
                plt.show()
            # """

            # Cut the observations to start at offset and end after year 3
            opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness = \
                lsst.select_observation_time_period(opsim_times, opsim_filters, opsim_psf, opsim_lim_mag,
                                                    opsim_sky_brightness, mjd_low=offset, mjd_high=61325)

            if len(opsim_times) == 0:
                continue

            obs_start = opsim_times[0]

            # Shift the observations back to the SN time frame
            opsim_times -= (obs_start - start_sn)

            # Perform nightly coadds
            opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness, N_coadds = \
                lsst.coadds(opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness)

            # Save all important properties
            time_series = []
            obs_days = []
            obs_filters = []
            obs_skybrightness = []
            obs_lim_mag = []
            obs_psf = []
            obs_N_coadds = []

            # Save the SN brightness
            model_mag = []  # apparent magnitude without scatter
            obs_mag = []  # apparent magnitude with scatter
            obs_mag_error = []
            app_mag_i_model = []
            obs_snr = []

            obs_mag_micro = []
            mag_micro_error = []
            obs_snr_micro = []
            app_mag_i_micro = []

            for observation in range(obs_upper_limit):

                if observation > len(opsim_times) - 1:
                    break

                day = opsim_times[observation]
                band = opsim_filters[observation]
                lim_mag = opsim_lim_mag[observation]
                psf = opsim_psf[observation]

                # For the r-filter, light curves with z > 1.6 are not defined. Skip these.
                # For the g-filter, light curves with z > 0.8 are not defined. Skip these.
                if band == 'r' and z_source > 1.6:
                    continue
                elif band == 'g' and z_source > 0.8:
                    continue
                elif band == 'u':
                    continue

                if day > end_sn:
                    break

                obs_days.append(day)
                obs_filters.append(band)
                obs_skybrightness.append(opsim_sky_brightness[observation])
                obs_lim_mag.append(lim_mag)
                obs_psf.append(psf)
                obs_N_coadds.append(N_coadds[observation])

                # Calculate microlensing contribution to light curve on this specific point in time
                if add_microlensing:
                    micro_day = microlensing.micro_snapshot(micro_contributions, td_images, day, band)
                    micro_day_i = microlensing.micro_snapshot(micro_contributions, td_images, day, 'i')
                else:
                    micro_day = np.nan
                    micro_day_i = np.nan

                # Calculate apparent magnitudes
                app_mag_model, app_mag_obs, app_mag_error, snr, app_mag_micro, app_mag_micro_error, \
                snr_micro = supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day, lsst, band,
                                                        lim_mag, add_microlensing)

                app_mag_model_i, app_mag_obs_i, _, _, \
                app_mag_obs_micro_i, _, _ = supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day_i,
                                                                        lsst, 'i', 24.0, add_microlensing)

                model_mag.append(np.array(app_mag_model))
                obs_mag.append(np.array(app_mag_obs))
                app_mag_i_model.append(np.array(app_mag_model_i))
                obs_mag_error.append(app_mag_error)
                obs_snr.append(snr)

                obs_mag_micro.append(np.array(app_mag_micro))
                mag_micro_error.append(np.array(app_mag_micro_error))
                obs_snr_micro.append(np.array(snr_micro))
                app_mag_i_micro.append(np.array(app_mag_obs_micro_i))

                # Calculate amplitude parameter
                amp_ps = lsst.app_mag_to_amplitude(app_mag_obs, band)

                """
                # Create the image and save it to the time-series list
                image_sim = lsst.generate_image(x_image, y_image, amp_ps, lens_model_class, source_model_class,
                                                lens_light_model_class, kwargs_lens, kwargs_source, kwargs_lens_light,
                                                band, psf)

                time_series.append(image_sim)
                """

                # _______________________________________________________________________

            obs_days = np.array(obs_days)
            obs_filters = np.array(obs_filters)
            obs_skybrightness = np.array(obs_skybrightness)
            obs_lim_mag = np.array(obs_lim_mag)
            obs_psf = np.array(obs_psf)
            obs_snr = np.array(obs_snr)
            obs_N_coadds = np.array(obs_N_coadds)

            model_mag = np.array(model_mag)
            obs_mag = np.array(obs_mag)
            app_mag_i_model = np.array(app_mag_i_model)
            obs_mag_error = np.array(obs_mag_error)

            obs_mag_micro = np.array(obs_mag_micro)
            mag_micro_error = np.array(mag_micro_error)
            obs_snr_micro = np.array(obs_snr_micro)
            app_mag_i_micro = np.array(app_mag_i_micro)

            obs_mag = obs_mag[:len(obs_days)]
            model_mag = model_mag[:len(obs_days)]

            # Final cuts

            # Determine whether the lensed SN is detectable, based on its brightness and flux ratio
            # if not supernova.check_detectability(lsst, model, macro_mag, brightness_im, obs_days_filters, micro_peak,
            #                                      add_microlensing):
            #     continue

            # Discard systems with fewer than obs_lower_limit images
            # L = len(time_series)
            # if L < obs_lower_limit:
            #     continue

            return obs_days, obs_filters, obs_skybrightness, obs_lim_mag, obs_psf, obs_snr, obs_N_coadds, model_mag, \
                   obs_mag, app_mag_i_model, obs_mag_error, obs_start, time_series, coords, Nobs_10yr, Nobs_3yr, \
                   obs_mag_micro, mag_micro_error, obs_snr_micro, app_mag_i_micro



def main():

    Simulations()


if __name__ == '__main__':
    main()