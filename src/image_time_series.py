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

# from microlensing.create_db import *


def simulate_time_series_images(num_samples, batch_size, batch, num_images, add_microlensing, obs_lower_limit,
                                obs_upper_limit, fixed_H0, lsst, Show, Save, path):

    """
    :param num_samples: total number of lens systems to be generated (int)
    :param batch_size: number of lens systems that is saved together in a batch (int)
    :param batch: the starting number of the batch
    :param num_images: number of lensed supernova images. choose between 2 (for doubles) and 4 (for quads)
    :param obs_times: list containing observation times for a sample of supernovae according to LSST Baseline v2.0
    :param obs_filters: list containing filters for a sample of supernovae according to LSST Baseline v2.0
    :param obs_all: list containing the observation times (mjd) of all objects together (in all bands)
    :param z_source_list_: array containing ~ 400,000 values of the source redshift
    :param z_lens_list_: array containing ~ 400,000 values of the lens redshift
    :param theta_E_list_: array containing ~ 400,000 values of the einstein radius
    :param add_microlensing: bool. if True: include microlensing effects
    :param telescope: telescope where the observations are modelled after. choose between 'LSST' and 'ZTF'
    :param bandpasses: bands/filters used for the lensed supernova observations
    :param obs_lower_limit: maximum number of observations (above which observations are cut off)
    :param obs_upper_limit: minimum number of observations (below which systems are discarded)
    :param fixed_H0: bool. if True: H0 is kept to a fixed value (evaluationsest). if False: H0 varies (training/test set)
    :param Show: bool. if True: figures and print statements show the properties of the lensed SN systems
    :param Save: bool. if True: output (time-series images and all properties) are saved in a pickle file
    :param path: only applies if Save=True. path where output is saved to
    :return: Generates image time-series and saves them to a pickle file
    """

    timer = Timer()
    timer.initiate('initiate')
    start_time = time.time()

    tqdm._instances.clear()
    pbar = tqdm(total=num_samples)
    counter = 0                     # Total number of attempts
    attempts = 0                    # Counts number of attempts per configuration
    sample_index = 0                # Counts how many configurations have been used (including failed ones)
    index = 0                       # Counts how many successful configurations have been used

    if batch_size > num_samples:
        print("Error: batch_size cannot be larger than num_samples!")
        sys.exit()

    if num_images != 2 and num_images != 4:
        print("Error: num_images should be equal to 2 (for doubles) or 4 (for quads)")
        sys.exit()



    # Change to small_sample=False!!!
    full_times, full_filters, full_skysig, full_zeropoint, full_ra, full_dec, full_MW_BV, full_psf = lsst.load_cadence(small_sample=True)
    full_times_all, _, _, _, _ = lsst.get_total_obs_times(full_times, full_filters)

    # Create telescope sky pointings
    ra_pointings, dec_pointings = lsst.create_sky_pointings(N=num_images)

    # Initialise OpSim Summary (SynOpSim) generator
    gen = lsst.generate_opsim_summary(ra_pointings, dec_pointings)

    # Create Pandas dataframe to store the data
    df = create_dataframe(batch_size)

    # Load joint theta_E, z_lens, z_source distribution
    z_source_list_, z_lens_list_, theta_E_list_ = lsst.load_z_theta(theta_min=0.05)

    # Sample num_samples from the joint z_lens, z_source, theta_E distribution
    # (Pick more samples since not all configurations will be successful)
    sample = np.random.choice(len(z_source_list_), size=10 * num_samples, replace=False)

    timer.end('initiate')

    while index < num_samples:

        timer.initiate('general_properties')
        # _______________________________________________________________________

        counter += 1
        attempts += 1

        # If tried more than 260 time unsucessfully; move on
        if attempts > 260:
            sample_index += 1
            attempts = 0
            continue

        # Sample lens configuration and cosmology
        z_source, z_lens, theta_E = lsst.sample_z_theta(z_source_list_, z_lens_list_, theta_E_list_,
                                                        sample, sample_index)
        if np.isnan(z_source):
            continue

        if fixed_H0:
            H_0 = 67.8  # Planck 2018 cosmology
        else:
            H_0 = np.random.uniform(20.0, 100.0)

        cosmo = FlatLambdaCDM(H0=H_0, Om0=0.315)
        time_delay_distance = get_time_delay_distance(z_source, z_lens, cosmo)
        source_x = np.random.uniform(-theta_E, theta_E)
        source_y = np.random.uniform(-theta_E, theta_E)

        timer.end('general_properties')
        timer.initiate('lens_SN_properties')
        # _______________________________________________________________________

        # Initiate the supernova and lens classes
        supernova = Supernova(theta_E, z_lens, z_source, cosmo, source_x, source_y)
        lens = Lens(theta_E, z_lens, z_source, cosmo)

        # _______________________________________________________________________

        # Lens specification
        lens_model_class, kwargs_lens, gamma_lens, e1_lens, e2_lens, gamma1, gamma2 = lens.mass_model(model='SIE')
        if np.isnan(gamma_lens):
            continue

        lens_light_model_class, kwargs_lens_light = lens.light_model()

        # Source specification (extended emission)
        source_model_class, kwargs_source = supernova.host_light_model()

        # Get image positions and magnifications
        x_image, y_image, macro_mag = supernova.get_image_pos_magnification(lens_model_class, kwargs_lens,
                                                                            min_distance=lsst.deltaPix,
                                                                            search_window=lsst.numPix * lsst.deltaPix)

        # Is num_images equal to 2 for doubles and to 4 for quads?
        if len(x_image) != num_images:
            continue

        # _______________________________________________________________________

        # Time delays between images (geometric + gravitational)
        td_images = lens.time_delays(lens_model_class, kwargs_lens, x_image, y_image)

        # Supernova light curve
        model, x1, c, MW_dust, M_B = supernova.light_curve(z_source)

        timer.end('lens_SN_properties')
        timer.initiate('detection_criteria_1')
        # _______________________________________________________________________

        # ---- Check image multiplicity method ----
        mult_method_peak = False

        sep = supernova.separation(x_image, y_image)

        # Is maximum image separation between 0.5 and 4.0 arcsec?
        if sep > 0.5 and sep < 4.0:
            # Check peak brightness and flux ratio: detectable?
            if supernova.check_detectability_peak(lsst, model, macro_mag, 0.0, False):
                mult_method_peak = True

        # ---- Check magnification method ----
        mag_method_peak = False

        # Sample different times to get the brightest combined flux
        time_range = np.linspace(min(td_images), max(td_images), 100)
        app_mag_i_unresolved = 50
        for t in time_range:
            app_mag_i_temp, _ = supernova.get_app_magnitude(model, t, macro_mag, td_images, np.nan, lsst.telescope,
                                                         'i', 27.79, 1, add_microlensing)
            app_mag_i_unresolved_temp = supernova.get_mags_unresolved(app_mag_i_temp, filler=None)

            if app_mag_i_unresolved_temp < app_mag_i_unresolved:
                app_mag_i_unresolved = app_mag_i_unresolved_temp

        M_i = model.source_peakabsmag(band='lssti', magsys='ab')
        mag_gap = -0.7

        # Checks whether eq. 1 from Wojtak et al. (2019) holds
        if app_mag_i_unresolved < M_i + cosmo.distmod(z_lens).value + mag_gap:
            mag_method_peak = True

        if not any([mult_method_peak, mag_method_peak]):
            continue

        timer.end('detection_criteria_1')
        # _______________________________________________________________________

        # Microlensing contributions

        if add_microlensing:

            timer.initiate('microlensing_1')

            microlensing = Microlensing(lens_model_class, kwargs_lens, x_image, y_image,
                                        theta_E, z_lens, z_source, cosmo, lsst.bandpasses)

            timer.end('microlensing_1')
            timer.initiate('microlensing_2')
            # _______________________________________________________________________

            # Convergence
            micro_kappa = microlensing.get_kappa()

            # Shear
            micro_gamma = microlensing.get_gamma()

            # Smooth-matter fraction
            _, R_eff = lens.scaling_relations()
            micro_s = microlensing.get_s(R_eff)
            if np.any(np.isnan(micro_s)):
                continue

            timer.end('microlensing_2')
            timer.initiate('microlensing_3')
            # _______________________________________________________________________

            # Load random microlensing light curves
            micro_lightcurves, macro_lightcurves, micro_times, mtiming1_, mtiming2_,\
                mmtiming1_, mmtiming2_, mmtiming3_, mmtiming4_, mmtiming5_, mmtiming6_, mmtiming7_ = microlensing.micro_lightcurve_all_images(micro_kappa,
                                                                                                         micro_gamma,
                                                                                                         micro_s)
            timer.end('microlensing_3')
            timer.initiate('microlensing_4')
            # _______________________________________________________________________

            # Calculate microlensing contribution at the peak
            micro_peak = microlensing.micro_snapshot(micro_lightcurves, macro_lightcurves, micro_times, td_images,
                                                     0, peak=True)

            # Check again if the peak brightness is detectable after microlensing
            if not supernova.check_detectability_peak(lsst, model, macro_mag, micro_peak, add_microlensing):
                continue

            timer.end('microlensing_4')
            # _______________________________________________________________________

        else:
            micro_kappa = np.nan
            micro_gamma = np.nan
            micro_s = np.nan
            micro_peak = 0.0
        # _______________________________________________________________________

        # Generate image time series

        # For this lens system, try 20 different observation sequences until it is detectable
        N_tries = 20
        for cadence_try in range(N_tries):

            timer.initiate('cadence')

            # Draw randomly an observation sequence
            obs_index = np.random.randint(0, len(full_times))
            # Start and end time of the lensed supernova
            start_sn = model.mintime() + min(td_images)
            end_sn = model.maxtime() + max(td_images)
            # Start the observations randomly between the start of 1st season and end of 3rd one
            offset = np.random.randint(min(full_times_all), max(full_times[obs_index]) + 2 * start_sn)
            # Get the observations that fall in that time period
            mask = [full_times[obs_index] > offset]
            days = full_times[obs_index][mask]
            filters = full_filters[obs_index][mask]
            coords = np.array([full_ra[obs_index], full_dec[obs_index]])
            obs_start = days[0]

            """
            if Show:
                plt.figure(5)
                plt.hist(obs_times[obs_index], bins=100)
                plt.axvline(x=days[0], color='C3')
                plt.show()
            """

            # Shift the observations back to the SN time frame
            days -= (offset - start_sn)

            time_series = []
            obs_days = []
            obs_filters = []
            obs_zeropoint = []
            obs_skysig = []
            obs_lim_mag = []

            # Keep track of the brightness of each observation
            obs_mag = np.ones((obs_upper_limit, len(x_image))) * 50
            obs_mag_error = []
            app_mag_i_obs = np.ones((obs_upper_limit, len(x_image))) * 50

            for observation in range(obs_upper_limit):

                if observation > len(days) - 1:
                    break

                day = days[observation]
                band = filters[observation]
                zeropoint = full_zeropoint[obs_index][mask][observation]
                skysig = full_skysig[obs_index][mask][observation]
                psf_sig = full_psf[obs_index][mask][observation]

                flux_skysig, lim_mag = lsst.get_weather(zeropoint, skysig, psf_sig)

                # For the r-filter, light curves with z > 1.5 are not defined. Skip these.
                if band == 'r' and z_source > 1.5:
                    continue

                if day > end_sn:
                    break

                obs_days.append(day)
                obs_filters.append(band)
                obs_zeropoint.append(zeropoint)
                obs_skysig.append(flux_skysig)
                obs_lim_mag.append(lim_mag)

                # Calculate microlensing contribution to light curve on this specific point in time
                if add_microlensing:
                    micro_day = microlensing.micro_snapshot(micro_lightcurves, macro_lightcurves, micro_times,
                                                            td_images, day)
                else:
                    micro_day = np.nan

                # Calculate apparent magnitudes
                app_mag_ps, app_mag_error = supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day,
                                                                        lsst.telescope, band, zeropoint, flux_skysig, add_microlensing)
                app_mag_ps_i, _ = supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day, lsst.telescope,
                                                           'i', zeropoint, flux_skysig, add_microlensing)

                obs_mag[observation] = np.array(app_mag_ps)
                app_mag_i_obs[observation] = np.array(app_mag_ps_i)
                obs_mag_error.append(app_mag_error)

                # Calculate amplitude parameter
                amp_ps = lsst.app_mag_to_amplitude(app_mag_ps, band)

                # Create the image and save it to the time-series list
                image_sim = lsst.generate_image(x_image, y_image, amp_ps, lens_model_class, source_model_class,
                                                lens_light_model_class, kwargs_lens, kwargs_source, kwargs_lens_light, band)

                time_series.append(image_sim)

                # _______________________________________________________________________


            obs_zeropoint = np.array(obs_zeropoint)
            obs_skysig = np.array(obs_skysig)
            obs_lim_mag = np.array(obs_lim_mag)
            obs_mag = obs_mag[:len(obs_days)]

            # Final cuts

            # Determine whether the lensed SN is detectable, based on its brightness and flux ratio
            # if not supernova.check_detectability(lsst, model, macro_mag, brightness_im, obs_days_filters, micro_peak,
            #                                      add_microlensing):
            #     continue

            # Discard systems with fewer than obs_lower_limit images
            L = len(time_series)
            # if L < obs_lower_limit:
            #     continue

            timer.end('cadence')

            break
            # _______________________________________________________________________

        timer.initiate('detection_criteria_2')

        try:
            obs_duration = obs_days[-1] - obs_days[0]
            obs_end = obs_start + obs_duration
        except:
            obs_end = obs_start

        # _______________________________________________________________________

        # Check detectability from observations

        # ---- Check image multiplicity method ----
        mult_method = False

        if sep > 0.5 and sep < 4.0:
            # Check maximum brightness and flux ratio: detectable?
            if supernova.check_detectability(lsst, model, macro_mag, obs_mag, obs_lim_mag, obs_filters, micro_peak,
                                                      add_microlensing):
                mult_method = True

        # ---- Check magnification method ----
        mag_method = False

        app_mag_i_obs_min = np.min(supernova.get_mags_unresolved(app_mag_i_obs, filler=50))

        if app_mag_i_obs_min < M_i + cosmo.distmod(z_lens).value + mag_gap:
            mag_method = True

        if Show:
            print("Theoretically visible with image multiplicity method?           ", mult_method_peak)
            print("Theoretically visible with magnification method?                ", mag_method_peak)
            print("Observations allow for detection with image multiplicity method?", mult_method)
            print("Observations allow for detection with magnification method?     ", mag_method)

        # Failed systems
        # if cadence_try == N_tries - 1:
        #     continue

        # The observations are detectable! Save the number of cadence tries it required
        # print("Detectable! Number of cadence tries: ", cadence_try + 1)

        timer.end('detection_criteria_2')
        timer.initiate('finalise')
        # _______________________________________________________________________

        # Cut out anything above obs_upper_limit observations
        if L > obs_upper_limit:
            del time_series[obs_upper_limit:]

        # Fill up time_series < obs_upper_limit with zero padding
        if L < obs_upper_limit:

            filler = np.zeros((48, 48))

            for i in range(obs_upper_limit - L):
                time_series.append(csr_matrix(filler))

        # Compute the maximum brightness in each bandpass
        obs_peak = supernova.brightest_obs_bands(lsst, macro_mag, obs_mag, obs_filters)

        # _______________________________________________________________________

        if Show:

            day_range = np.linspace(min(td_images) - 100, max(td_images) + 100, 250)

            if add_microlensing:
                micro_day_range = []
                for d in day_range:
                    micro_day_range.append(
                        np.array(microlensing.micro_snapshot(micro_lightcurves, macro_lightcurves, micro_times,
                                                             td_images, d)))
                micro_day_range = np.array(micro_day_range)
            else:
                micro_day_range = np.nan


            sigma_bkg_i = lsst.single_band_properties('i')[2]
            data_class_i = lsst.grid(sigma_bkg_i)[0]

            visualise = Visualisation(time_delay_distance, td_images, theta_E, data_class_i, macro_mag, obs_days, obs_filters)

            # Print the properties of the lensed supernova system
            visualise.print_properties(z_lens, z_source, H_0, micro_peak, obs_peak)

            # Plot time delay surface
            visualise.plot_td_surface(lens_model_class, kwargs_lens, source_x, source_y, x_image, y_image)

            # Plot light curve with observation epochs
            visualise.plot_light_curves(model, day_range, micro_day_range, add_microlensing, obs_mag, obs_mag_error)
            visualise.plot_light_curves_perband(model, day_range, micro_day_range, add_microlensing, obs_mag, obs_mag_error)

            # Display all observations:
            visualise.plot_observations(time_series)

        # ____________________________________________________________________________

        obs_mag_unresolved = supernova.get_mags_unresolved(obs_mag)

        # Save the desired quantities in the data frame
        df = write_to_df(df, index, batch_size, time_series, z_source, z_lens, H_0, theta_E, obs_peak, obs_days,
                         obs_filters, obs_mag, obs_mag_error, obs_mag_unresolved, macro_mag, source_x, source_y,
                         td_images, time_delay_distance, x_image, y_image, gamma_lens, e1_lens, e2_lens, days, gamma1,
                         gamma2, micro_kappa, micro_gamma, micro_s, micro_peak, x1, c, M_B, obs_start, obs_end,
                         mult_method_peak, mult_method, mag_method_peak, mag_method, coords, obs_zeropoint, obs_skysig,
                         obs_lim_mag)

        # Check if the data frame is full
        if (index+1) % batch_size == 0 and index > 1:
            if Save:
                # Save data frame to laptop
                df.to_pickle(path + "Baselinev20_new_weather_numimages=" + str(int(num_images)) + "_batch" + str(str(batch).zfill(3)) + ".pkl")

            if (index+1) < num_samples:
                # Start a new, empty data frame
                df = create_dataframe(batch_size)
            batch += 1

        # Update variables
        sample_index += 1
        index += 1
        pbar.update(1)
        attempts = 0
        rejected_cadence = 0
        accepted_peak = 0

        timer.end('finalise')
        # _______________________________________________________________________

    end_time = time.time()
    duration = end_time - start_time

    print("Done!")
    print("Simulating images took ", np.around(duration), "seconds (", np.around(duration / 3600, 2), "hours) to complete.")
    print("Number of image-time series generated: ", index)
    print("Number of configurations tried: ", sample_index)
    print("Number of attempts: ", counter)
    print(" ")
    print(df)

    return df, timer.timing_dict


def main():
    telescope = 'LSST'
    bandpasses = ['r', 'i', 'z', 'y']
    num_samples = 1           # Total number of lens systems to be generated
    batch_size = 1            # Number of lens systems that is saved together in a batch
    batch = 1                 # Starting number of the batch
    num_images = 2            # Choose between 2 (for doubles) and 4 (for quads)
    obs_upper_limit = 40      # Upper limit of number of observations
    obs_lower_limit = 5       # Lower limit of number of observations
    fixed_H0 = True           # Bool, if False: vary H0. if True: fix H0 to 70 km/s/Mpc (for the evaluation set)
    add_microlensing = False  # Bool, if False: Only macro magnification. if True: Add effects of microlensing

    Show = True               # Bool, if True: Show figures and print information about the lens systems
    Save = False              # Bool, if True: Save image time-series
    path = "../processed_data/Cadence_microlensing_evaluationset_doubles2/"  # Path to folder in which to save the results

    lsst = Telescope(telescope, bandpasses)
    z_source_list_, z_lens_list_, theta_E_list_ = lsst.load_z_theta(theta_min=0.1)


if __name__ == '__main__':
    main()

