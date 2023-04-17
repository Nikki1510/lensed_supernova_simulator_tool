#! /bin/python3
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
# from tqdm import tqdm
from tqdm import notebook
from tqdm.auto import tqdm
from astropy.cosmology import FlatLambdaCDM
from scipy.sparse import csr_matrix
from functions import create_dataframe_unlensed, write_to_df_unlensed
from class_supernova_unlensed import Supernova_Unlensed
from class_visualisation_unlensed import Visualisation_Unlensed
from class_telescope import Telescope
from class_timer import Timer


def simulate_unlensed_sne(num_samples, batch_size, batch, obs_lower_limit,
                                obs_upper_limit, fixed_H0, lsst, Show, Save, path):

    """
    :param num_samples: total number of lens systems to be generated (int)
    :param batch_size: number of lens systems that is saved together in a batch (int)
    :param batch: the starting number of the batch
    :param obs_lower_limit: maximum number of observations (above which observations are cut off)
    :param obs_upper_limit: minimum number of observations (below which systems are discarded)
    :param fixed_H0: bool. if True: H0 is kept to a fixed value (evaluationsest). if False: H0 varies (training/test set)
    :param lsst: telescope where the observations are modelled after. choose between 'LSST' and 'ZTF'
    :param Show: bool. if True: figures and print statements show the properties of the lensed SN systems
    :param Save: bool. if True: output (time-series images and all properties) are saved in a pickle file
    :param path: only applies if Save=True. path where output is saved to
    :return: Generates light curves & image time-series and saves them to a pickle file
    """

    timer = Timer()
    timer.initiate('initiate')
    start_time = time.time()

    tqdm._instances.clear()
    pbar = tqdm(total=num_samples, position=0, leave=True) # notebook.tqdm
    counter = 0                     # Total number of attempts
    attempts = 0                    # Counts number of attempts per configuration
    sample_index = 0                # Counts how many configurations have been used (including failed ones)
    index = 0                       # Counts how many successful configurations have been used

    if batch_size > num_samples:
        print("Error: batch_size cannot be larger than num_samples!")
        sys.exit()

    # Get OpSim Summary generator
    gen = lsst.gen

    # Create Pandas dataframe to store the data
    df = create_dataframe_unlensed(batch_size)

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

        if fixed_H0:
            H_0 = 67.8  # Planck 2018 cosmology
        else:
            H_0 = np.random.uniform(20.0, 100.0)

        cosmo = FlatLambdaCDM(H0=H_0, Om0=0.315)

        timer.end('general_properties')
        timer.initiate('lens_SN_properties')
        # _______________________________________________________________________

        # Initiate the unlensed supernova class
        supernova = Supernova_Unlensed(cosmo)
        z_source = supernova.z_source

        # Supernova light curve
        model, x1, c, MW_dust, M_B = supernova.light_curve(z_source)

        timer.end('lens_SN_properties')
        timer.initiate('detection_criteria_1')
        # _______________________________________________________________________

        # ---- Check peak detectability (in i-band) ----

        lim_mag_i = lsst.single_band_properties('i')[1]
        app_mag_i = supernova.get_app_magnitude(model, 0.0, lsst, 'i', lim_mag_i)[0]

        if app_mag_i > lim_mag_i:
            continue

        timer.end('detection_criteria_1')
        # _______________________________________________________________________

        # Generate image time series

        # Here, you can define a maximum number of tries.
        N_tries = 20

        for cadence_try in range(N_tries):

            timer.initiate('cadence')

            ra, dec, opsim_times, opsim_filters, opsim_psf, opsim_lim_mag, opsim_sky_brightness = lsst.opsim_observation(gen)

            coords = np.array([ra, dec])

            # Start and end time of the lensed supernova
            start_sn = model.mintime()
            end_sn = model.maxtime()

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

                # Calculate apparent magnitudes
                app_mag_model, app_mag_obs, app_mag_error, snr = supernova.get_app_magnitude(model, day, lsst, band, lim_mag)
                app_mag_model_i, app_mag_obs_i, _, _ = supernova.get_app_magnitude(model, day, lsst, 'i', 24.0)

                model_mag.append(np.array(app_mag_model))
                obs_mag.append(np.array(app_mag_obs))
                app_mag_i_model.append(np.array(app_mag_model_i))
                obs_mag_error.append(app_mag_error)
                obs_snr.append(snr)

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

            timer.end('cadence')

            break
            # _______________________________________________________________________

        try:
            obs_duration = obs_days[-1] - obs_days[0]
            obs_end = obs_start + obs_duration
        except:
            obs_end = obs_start

        timer.initiate('finalise')
        # _______________________________________________________________________

        # Compute the maximum brightness in each bandpass
        obs_peak = supernova.brightest_obs_bands(lsst, obs_mag, obs_filters)

        # _______________________________________________________________________

        if Show:

            day_range = np.linspace(-50, 100, 250)

            sigma_bkg_i = lsst.single_band_properties('i')[2]
            data_class_i = lsst.grid(sigma_bkg_i)[0]

            visualise = Visualisation_Unlensed(data_class_i, obs_days, obs_filters)

            # Print the properties of the lensed supernova system
            visualise.print_properties(z_source, H_0, obs_peak)

            # Plot light curve with observation epochs
            visualise.plot_light_curves(model, day_range, model_mag)
            visualise.plot_light_curves_perband(model, day_range, model_mag, obs_mag, obs_mag_error)

        # ____________________________________________________________________________

        # Save the desired quantities in the data frame
        df = write_to_df_unlensed(df, index, batch_size, z_source, H_0, obs_peak, obs_days,
                         obs_filters, model_mag, obs_mag, obs_mag_error, obs_snr, x1, c, M_B, obs_start, obs_end,
                         coords, obs_skybrightness, obs_psf, obs_lim_mag, obs_N_coadds)

        # Check if the data frame is full
        if (index+1) % batch_size == 0 and index > 1:
            if Save:
                # Save data frame to laptop
                df.to_pickle(path + "Baselinev30_unlensed_newrates_batch" + str(str(batch).zfill(3)) + ".pkl")

            if (index+1) < num_samples:
                # Start a new, empty data frame
                df = create_dataframe_unlensed(batch_size)
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


if __name__ == '__main__':
    main()

