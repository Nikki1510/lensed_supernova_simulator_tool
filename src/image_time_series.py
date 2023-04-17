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
    :param add_microlensing: bool. if True: include microlensing effects
    :param lsst: telescope where the observations are modelled after. choose between 'LSST' and 'ZTF'
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
    pbar = tqdm(total=num_samples, position=0, leave=True)
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

    # Get OpSim Summary generator
    gen = lsst.gen

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
            lim_mag_i = lsst.single_band_properties('i')[1]
            app_mag_i_temp, _, _, _ = supernova.get_app_magnitude(model, t, macro_mag, td_images, np.nan, lsst,
                                                         'i', lim_mag_i, add_microlensing)
            app_mag_i_unresolved_temp = supernova.get_mags_unresolved(app_mag_i_temp, lsst, ['i'], 24.0, filler=None)[0]

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

        # Here, you can define a maximum number of tries.
        N_tries = 20
        for cadence_try in range(N_tries):

            timer.initiate('cadence')

            ra, dec, opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness = lsst.opsim_observation(gen)

            coords = np.array([ra, dec])

            # Start and end time of the lensed supernova
            start_sn = model.mintime() + min(td_images)
            end_sn = model.maxtime() + max(td_images)

            # Start the SN randomly between the start of 1st season and end of 3rd one (dates from Catarina)
            offset = np.random.randint(60220, 61325-50)

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
            model_mag = []     # apparent magnitude without scatter
            obs_mag = []      # apparent magnitude with scatter
            obs_mag_error = []
            app_mag_i_model = []
            obs_snr = []

            for observation in range(obs_upper_limit):

                if observation > len(opsim_times) - 1:
                    break

                day = opsim_times[observation]
                band = opsim_filters[observation]
                lim_mag = opsim_lim_mag[observation]


                # For the r-filter, light curves with z > 1.5 are not defined. Skip these.
                # !! Also do this for u and g bands !!
                if band == 'r' and z_source > 1.5:
                    continue
                elif band == 'g':
                    continue
                elif band == 'u':
                    continue

                if day > end_sn:
                    break

                obs_days.append(day)
                obs_filters.append(band)
                obs_skybrightness.append(opsim_sky_brightness[observation])
                obs_lim_mag.append(lim_mag)
                obs_psf.append(opsim_psf[observation])
                obs_N_coadds.append(N_coadds[observation])


                # Calculate microlensing contribution to light curve on this specific point in time
                if add_microlensing:
                    micro_day = microlensing.micro_snapshot(micro_lightcurves, macro_lightcurves, micro_times,
                                                            td_images, day)
                else:
                    micro_day = np.nan

                # Calculate apparent magnitudes
                app_mag_model, app_mag_obs, app_mag_error, snr = supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day,
                                                                        lsst, band, lim_mag, add_microlensing)
                app_mag_model_i, app_mag_obs_i, _, _ = supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day, lsst,
                                                           'i', 24.0, add_microlensing)

                model_mag.append(np.array(app_mag_model))
                obs_mag.append(np.array(app_mag_obs))
                app_mag_i_model.append(np.array(app_mag_model_i))
                obs_mag_error.append(app_mag_error)
                obs_snr.append(snr)

                # Calculate amplitude parameter
                amp_ps = lsst.app_mag_to_amplitude(app_mag_obs, band)

                # Create the image and save it to the time-series list
                image_sim = lsst.generate_image(x_image, y_image, amp_ps, lens_model_class, source_model_class,
                                                lens_light_model_class, kwargs_lens, kwargs_source, kwargs_lens_light, band)

                time_series.append(image_sim)

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

            obs_mag = obs_mag[:len(obs_days)]
            model_mag = model_mag[:len(obs_days)]

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

        app_mag_i_obs_min = np.min(supernova.get_mags_unresolved(app_mag_i_model, lsst,
                                                                 ['i' for i in range(len(app_mag_i_model))],
                                                                 [24 for i in range(len(app_mag_i_model))],
                                                                 filler=np.nan)[0])

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
            visualise.plot_light_curves_perband(model, day_range, micro_day_range, add_microlensing, model_mag, obs_mag, obs_mag_error)

            # Display all observations:
            visualise.plot_observations(time_series)

        # ____________________________________________________________________________

        obs_mag_unresolved, mag_unresolved_error, snr_unresolved = supernova.get_mags_unresolved(obs_mag, lsst, obs_filters, obs_lim_mag)

        # Save the desired quantities in the data frame
        df = write_to_df(df, index, batch_size, time_series, z_source, z_lens, H_0, theta_E, obs_peak, obs_days,
                         obs_filters, model_mag, obs_mag, obs_mag_error, obs_snr, obs_mag_unresolved, mag_unresolved_error,
                         snr_unresolved, macro_mag, source_x, source_y,
                         td_images, time_delay_distance, x_image, y_image, gamma_lens, e1_lens, e2_lens, gamma1,
                         gamma2, micro_kappa, micro_gamma, micro_s, micro_peak, x1, c, M_B, obs_start, obs_end,
                         mult_method_peak, mult_method, mag_method_peak, mag_method, coords, obs_skybrightness, obs_psf,
                         obs_lim_mag, obs_N_coadds)

        # Check if the data frame is full
        if (index+1) % batch_size == 0 and index > 1:
            if Save:
                # Save data frame to laptop
                df.to_pickle(path + "Baselinev30_coadds_numimages=" + str(int(num_images)) + "_batch" + str(str(batch).zfill(3)) + ".pkl")

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
    # ---------------------------------
    telescope = 'LSST'
    bandpasses = ['r', 'i', 'z', 'y']
    num_samples = 20
    # ---------------------------------

    lsst = Telescope(telescope, bandpasses, num_samples)

    num_samples = 1  # Total number of lens systems to be generated
    batch_size = 1  # Number of lens systems that is saved together in a batch
    batch = 1  # Starting number of the batch
    num_images = 2  # Choose between 2 (for doubles) and 4 (for quads)
    obs_upper_limit = 100  # Upper limit of number of observations
    obs_lower_limit = 5  # Lower limit of number of observations
    fixed_H0 = True  # Bool, if False: vary H0. if True: fix H0 to 70 km/s/Mpc (for the evaluation set)
    add_microlensing = False  # Bool, if False: Only macro magnification. if True: Add effects of microlensing

    Show = False  # Bool, if True: Show figures and print information about the lens systems
    Save = False  # Bool, if True: Save image time-series
    path = "../processed_data/Baseline_v_2_0_/"  # Path to folder in which to save the results

    df, timings = simulate_time_series_images(num_samples, batch_size, batch, num_images, add_microlensing,
                                              obs_lower_limit, obs_upper_limit, fixed_H0, lsst, Show, Save, path)


    """
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
    """

if __name__ == '__main__':
    main()

