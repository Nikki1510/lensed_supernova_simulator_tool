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

    def initialise_parameters(self, lsst, z_source_list_, z_lens_list_, theta_E_list_, sample, sample_index, fixed_H0,
                              num_images, attempts):
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
        :param num_images: number of lensed supernova images. choose between 2 (for doubles) and 4 (for quads)
        :param attempts: counts number of attempts per configuration
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

        if num_images == 2:
            source_x = np.random.uniform(-theta_E, theta_E)
            source_y = np.random.uniform(-theta_E, theta_E)
        elif num_images == 4:
            source_x = np.random.uniform(-0.4 * theta_E, 0.4 * theta_E)
            source_y = np.random.uniform(-0.4 * theta_E, 0.4 * theta_E)

        return z_source, z_lens, theta_E, H_0, cosmo, time_delay_distance, source_x, source_y

    def check_mult_method_peak(self, supernova, sep, lsst, model, macro_mag, td_images, add_microlensing, microlensing, micro_contributions):
        """
        Check whether multiple images of the lensed SN are visible (evaluated at peak brightness for each image).

        :param supernova: class that contains supernova functions
        :param sep: maximum separation between the SN images
        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param model: SNcosmo model for the supernova light curve
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param micro_peak: array of length [num_images] containing the microlensing contributions at peak
        :return: mult_method_peak: bool. if True: lensed SN passes the image multiplicity method
        """

        mult_method_peak = False

        # Is maximum image separation between 0.5 and 4.0 arcsec?
        if sep > 0.5 and sep < 4.0:

            # Check peak brightness and flux ratio: detectable?
            if supernova.check_detectability_peak(lsst, model, macro_mag, td_images, add_microlensing, microlensing, micro_contributions):
                mult_method_peak = True

        return mult_method_peak

    def check_mag_method_peak(self, td_images, lsst, supernova, model, macro_mag, z_source, m_lens, mag_gap,
                              add_microlensing, microlensing, micro_contributions):
        """
        Check whether the unresolved supernova images are brighter than a typical type Ia SN at the lens redshift.

        :param td_images: array of length [num_images] containing the relative time delays between the supernova images
        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param supernova: class that contains supernova functions
        :param model: SNcosmo model for the supernova light curve
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param z_source: redshift of the supernova (float)
        :param m_lens: array with for each band the apparent magnitude of a vanilla type Ia at the lens redshift
        :param mag_gap: number of magnitudes the lensed SN must be brighter than a typical Ia at the lens redshift
        :param add_microlensing: bool. if False: no microlensing. if True: also add microlensing to the peak
        :param microlensing: microlensing class
        :param micro_contributions: list of length [num_images] containing microlensing dictionaries
        :return: mag_method_peak: bool. if True: lensed SN passes the magnification method,
                 peak_magnitudes: array containing the unresolved peak apparent magnitudes in each band
        """

        mag_method_peak = False

        # Sample different times to get the brightest combined flux
        time_range = np.linspace(min(td_images)-10, max(td_images), 200)

        # Save peak magnitudes in each band
        peak_magnitudes = []

        for band in lsst.bandpasses:

            if band == 'r' and z_source > 1.6:
                peak_magnitudes.append(np.nan)
                continue
            elif band == 'g' and z_source > 0.8:
                peak_magnitudes.append(np.nan)
                continue

            app_mag_unresolved = 50

            for t in time_range:
                lim_mag_band = lsst.single_band_properties(band)[4]

                if add_microlensing:
                    micro_day = microlensing.micro_snapshot(micro_contributions, td_images, t, band)
                    app_mag_model = supernova.get_app_magnitude(model, t, macro_mag, td_images, micro_day, lsst, band,
                                                           lim_mag_band, add_microlensing)[7]
                else:
                    micro_day = np.nan
                    app_mag_model = supernova.get_app_magnitude(model, t, macro_mag, td_images, micro_day, lsst, band,
                                                                lim_mag_band, add_microlensing)[1]

                app_mag_unresolved_temp = supernova.get_mags_unresolved(app_mag_model, lsst, [band], lim_mag_band, filler=None)[0]

                if app_mag_unresolved_temp < app_mag_unresolved:
                    app_mag_unresolved = app_mag_unresolved_temp

            # Checks whether eq. 1 from Wojtak et al. (2019) holds
            if app_mag_unresolved < m_lens[band] + mag_gap:
                if app_mag_unresolved < lim_mag_band:
                    mag_method_peak = True

            # if app_mag_unresolved < M + cosmo.distmod(z_lens).value + mag_gap:
            #    mag_method_peak = True

            #print("band: ", band)
            #print("app_mag_min: ", app_mag_unresolved)
            #print("comparison: ", m_lens[band] + mag_gap)
            #print("detected? ", mag_method_peak)
            #print(" ")

            peak_magnitudes.append(app_mag_unresolved)

        return mag_method_peak, np.array(peak_magnitudes)

    def check_mult_method(self, supernova, sep, lsst, model, macro_mag, obs_mag, obs_snr, obs_lim_mag, obs_filters, micro_peak,
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
        :param obs_snr: array of shape [N_observations, N_images] containing the S/N ratio for each image
        :param obs_lim_mag: array of length N_observations containing the limiting magnitude (5 sigma depth)
        :param obs_filters: array of length N_observations containing the bandpasses for each observation
        :param micro_peak: array of length [num_images] containing the microlensing contributions at light curve peak
        :param add_microlensing: bool. if False: no microlensing. if True: also add microlensing to the peak
        :return: mult_method: bool. if True: lensed SN passes the image multiplicity method
        """

        mult_method = False

        if sep > 0.5 and sep < 4.0:

            # Check maximum brightness and flux ratio: detectable?
            if supernova.check_detectability(lsst, model, macro_mag, obs_mag, obs_snr, obs_lim_mag, obs_filters, micro_peak,
                                             add_microlensing):
                mult_method = True

        return mult_method

    def check_mag_method(self, app_mag_unresolved, snr_unresolved, obs_filters, lsst, cosmo, z_lens, z_source, m_lens, mag_gap):
        """
        Check whether the unresolved supernova images are brighter than a typical type Ia SN at the lens redshift.

        :param app_mag_unresolved: array of shape [N_observations] that contains the unresolved apparent magnitudes
        :param snr_unresolved: array of shape [N_observations] that contains the S/N ratio of the unresolved observations
        :param obs_filters: array of length N_observations containing the bandpasses for each observation
        :param lsst: telescope class where the observations are modelled after. choose between 'LSST' and 'ZTF'
        :param cosmo: instance of astropy containing the background cosmology
        :param z_lens: redshift of the lens galaxy (float)
        :param z_source: redshift of the supernova (float)
        :param m_lens: array with for each band the apparent magnitude of a vanilla type Ia at the lens redshift
        :param mag_gap: number of magnitudes the lensed SN must be brighter than a typical Ia at the lens redshift
        :return: mag_method: bool. if True: lensed SN passes the magnification method
        """

        for band in lsst.bandpasses:

            if band == 'r' and z_source > 1.6:
                continue
            elif band == 'g' and z_source > 0.8:
                continue

            mask = np.where((obs_filters == band) & (np.isfinite(app_mag_unresolved)))
            if len(app_mag_unresolved[mask]) == 0:
                continue

            # Find minimum unresolved magnitude (and corresponding S/N ratio)
            app_mag_min_index = np.argmin(app_mag_unresolved[mask])
            app_mag_min = app_mag_unresolved[mask][app_mag_min_index]
            snr_min = snr_unresolved[mask][app_mag_min_index]

            #print("band: ", band)
            #print("app_mag_min: ", app_mag_min)
            #print("comparison: ", m_lens[band] + mag_gap)
            #print("detected? ", app_mag_min < m_lens[band] + mag_gap)
            #print("snr: ", snr_min)
            #print(" ")

            # Check if it passes the magnification method condition
            if app_mag_min < m_lens[band] + mag_gap:
                # Is the S/N ratio high enough for a detection?
                if snr_min >= 5.0:
                    return True

        return False


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
                                                    opsim_sky_brightness, mjd_low=offset, mjd_high=61325)  # !!! Remove: change back: mjd_low=offset, mjd_high=61325
            # Testing: mjd_low=60220, mjd_high=63325

            if len(opsim_times) == 0:
                continue

            obs_start = opsim_times[0]

            # Shift the observations back to the SN time frame
            opsim_times -= (obs_start - start_sn)

            # Testing: comment out
            # Perform nightly coadds
            opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness, N_coadds = \
                lsst.coadds(opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness)

            # Testing: uncomment
            #if len(opsim_filters) < 300:  # !!! Remove !!!
            #    continue

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

            # !!! Remove !!! Testing:
            #opsim_times = np.linspace(opsim_times[0], opsim_times[0]+100, 400)
            #opsim_filters = opsim_filters[:len(opsim_times)]  # ['g', 'r', 'i', 'z', 'y'] * 80  # ['i' for i in range(len(opsim_times))]

            #opsim_lim_mag = opsim_lim_mag[:len(opsim_times)]

            #opsim_sky_brightness = [26 for i in range(len(opsim_times))]
            #opsim_psf = [24 for i in range(len(opsim_times))]
            #N_coadds = [24 for i in range(len(opsim_times))]
            # !!! Remove !!!

            for observation in range(obs_upper_limit):

                if observation > len(opsim_filters) - 1:
                    break

                day = opsim_times[observation]
                band = opsim_filters[observation]
                lim_mag = opsim_lim_mag[observation]
                psf = opsim_psf[observation]

                # For the r-filter, light curves with z > 1.6 are not defined. Skip these.
                # For the g-filter, light curves with z > 0.8 are not defined. Skip these.
                if band == 'r' and z_source > 1.6:
                    # continue
                    filling = np.ones(len(td_images)) * np.nan
                    app_mag_model = app_mag_obs = app_mag_error = snr = app_mag_micro = app_mag_micro_error = snr_micro = filling
                    app_mag_model_i = app_mag_model_i_micro = app_mag_obs_micro_i = filling

                elif band == 'g' and z_source > 0.8:
                    # continue
                    filling = np.ones(len(td_images)) * np.nan
                    app_mag_model = app_mag_obs = app_mag_error = snr = app_mag_micro = app_mag_micro_error = snr_micro = filling
                    app_mag_model_i = app_mag_model_i_micro = app_mag_obs_micro_i = filling

                elif band == 'u':
                    continue
                else:
                    # Calculate microlensing contribution to light curve on this specific point in time
                    if add_microlensing:
                        micro_day = microlensing.micro_snapshot(micro_contributions, td_images, day, band)
                        micro_day_i = microlensing.micro_snapshot(micro_contributions, td_images, day, 'i')
                    else:
                        micro_day = np.nan
                        micro_day_i = np.nan

                    # Calculate apparent magnitudes
                    app_mag_model, app_mag_obs, app_mag_error, snr, app_mag_micro, app_mag_micro_error, \
                    snr_micro, _ = supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day, lsst, band,
                                                            lim_mag, add_microlensing)

                    app_mag_model_i, app_mag_obs_i, _, _, app_mag_obs_micro_i, _, _, app_mag_model_i_micro = \
                        supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day_i, lsst, 'i', 24.0, add_microlensing)

                if day > end_sn:
                    break

                obs_days.append(day)
                obs_filters.append(band)
                obs_skybrightness.append(opsim_sky_brightness[observation])
                obs_lim_mag.append(lim_mag)
                obs_psf.append(psf)
                obs_N_coadds.append(N_coadds[observation])

                model_mag.append(np.array(app_mag_model))
                obs_mag.append(np.array(app_mag_obs))
                app_mag_i_model.append(np.array(app_mag_model_i))
                obs_mag_error.append(app_mag_error)
                obs_snr.append(snr)

                obs_mag_micro.append(np.array(app_mag_micro))
                mag_micro_error.append(np.array(app_mag_micro_error))
                obs_snr_micro.append(np.array(snr_micro))
                app_mag_i_micro.append(np.array(app_mag_model_i_micro))

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