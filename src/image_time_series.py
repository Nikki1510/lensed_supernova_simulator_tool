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
from class_simulations import Simulations

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

    simulations = Simulations()

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

        z_source, z_lens, theta_E, H_0, cosmo, time_delay_distance, source_x, source_y = \
            simulations.initialise_parameters(lsst, z_source_list_, z_lens_list_, theta_E_list_, sample, sample_index,
                                              fixed_H0)

        if np.isnan(z_source):
            continue

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

        # Check image multiplicity method
        sep = supernova.separation(x_image, y_image)
        mult_method_peak = simulations.check_mult_method_peak(supernova, sep, lsst, model, macro_mag)

        # Check magnification method
        M_i = model.source_peakabsmag(band='lssti', magsys='ab')
        mag_gap = -0.7
        mag_method_peak = simulations.check_mag_method_peak(td_images, lsst, supernova, model, macro_mag,
                                                            add_microlensing, cosmo, z_lens, M_i, mag_gap)

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
            micro_contributions = microlensing.micro_lightcurve_all_images(micro_kappa, micro_gamma, micro_s)

            timer.end('microlensing_3')
            timer.initiate('microlensing_4')
            # _______________________________________________________________________

            # Calculate microlensing contribution at the peak
            micro_peak = microlensing.micro_snapshot(micro_contributions, td_images, 0, 'i', peak=True)

            # Check again if the peak brightness is detectable after microlensing
            # if not supernova.check_detectability_peak(lsst, model, macro_mag, micro_peak, add_microlensing):
            #    continue

            timer.end('microlensing_4')
            # _______________________________________________________________________

        else:
            micro_kappa = np.nan
            micro_gamma = np.nan
            micro_s = np.nan
            micro_peak = 0.0
            microlensing = np.nan
            micro_contributions = np.ones(len(x_image)) * np.nan
        # _______________________________________________________________________

        # Perform the observations: generate image time series and light curves

        timer.initiate('cadence')

        obs_days, obs_filters, obs_skybrightness, obs_lim_mag, obs_psf, obs_snr, obs_N_coadds, model_mag, obs_mag, \
        app_mag_i_model, obs_mag_error, obs_start, time_series, coords, Nobs_10yr, Nobs_3yr, obs_mag_micro, \
        mag_micro_error, obs_snr_micro, app_mag_i_micro = \
            simulations.get_observations(lsst, supernova, gen, model, td_images, x_image, y_image, z_source, macro_mag,
                                         lens_model_class, source_model_class, lens_light_model_class, kwargs_lens,
                                         kwargs_source, kwargs_lens_light, add_microlensing, microlensing,
                                         micro_contributions, obs_upper_limit, Show)

        L = len(time_series)

        timer.end('cadence')

        # _______________________________________________________________________

        timer.initiate('detection_criteria_2')

        try:
            obs_duration = obs_days[-1] - obs_days[0]
            obs_end = obs_start + obs_duration
        except:
            obs_end = obs_start

        # _______________________________________________________________________

        # Check detectability from observations

        # Check image multiplicity method
        mult_method = simulations.check_mult_method(supernova, sep, lsst, model, macro_mag, obs_mag, obs_lim_mag,
                                                    obs_filters, 0.0, add_microlensing=False)
        mult_method_micro = simulations.check_mult_method(supernova, sep, lsst, model, macro_mag, obs_mag_micro, obs_lim_mag,
                                                    obs_filters, micro_peak, add_microlensing)

        # Check magnification method
        mag_method = simulations.check_mag_method(supernova, app_mag_i_model, lsst, M_i, cosmo, z_lens, mag_gap)
        mag_method_micro = simulations.check_mag_method(supernova, app_mag_i_micro, lsst, M_i, cosmo, z_lens, mag_gap)

        if Show:
            print("Theoretically visible with image multiplicity method?           ", mult_method_peak)
            print("Theoretically visible with magnification method?                ", mag_method_peak)
            print("Observations allow for detection with image multiplicity method?", mult_method)
            print("Observations allow for detection with magnification method?     ", mag_method)
            print("Microlensing allow for detection with image multiplicity method?", mult_method_micro)
            print("Microlensing allow for detection with magnification method?     ", mag_method_micro)

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

        # Compute unresolved brightness
        obs_mag_unresolved, mag_unresolved_error, snr_unresolved = supernova.get_mags_unresolved(obs_mag, lsst,
                                                                                                 obs_filters,
                                                                                                 obs_lim_mag)
        if add_microlensing:
            mag_unresolved_micro, mag_unresolved_micro_error, \
            snr_unresolved_micro = supernova.get_mags_unresolved(obs_mag_micro, lsst, obs_filters, obs_lim_mag)
        else:
            mag_unresolved_micro, mag_unresolved_micro_error, snr_unresolved_micro = np.nan, np.nan, np.nan

        # Determine if the observation belongs to the WFD/DDF/galactic plane
        survey, rolling = lsst.determine_survey(Nobs_3yr, Nobs_10yr, obs_start)

        # _______________________________________________________________________

        if Show:

            day_range = np.linspace(min(td_images) - 100, max(td_images) + 100, 250)

            sigma_bkg_i = lsst.single_band_properties('i')[2]
            data_class_i = lsst.grid(sigma_bkg_i)[0]

            visualise = Visualisation(time_delay_distance, td_images, theta_E, data_class_i, macro_mag, obs_days, obs_filters)

            # Print the properties of the lensed supernova system
            visualise.print_properties(z_lens, z_source, H_0, micro_peak, obs_peak)

            # Plot time delay surface
            visualise.plot_td_surface(lens_model_class, kwargs_lens, source_x, source_y, x_image, y_image)

            # Plot light curve with observation epochs
            visualise.plot_light_curves(model, day_range, obs_mag_unresolved, mag_unresolved_error)
            visualise.plot_light_curves_perband(z_source, model, day_range, model_mag, obs_mag, obs_mag_error, add_microlensing,
                                                microlensing, micro_contributions)
            if add_microlensing:
                visualise.plot_light_curves_perband(z_source, model, day_range, model_mag, obs_mag_micro,
                                                    mag_micro_error, add_microlensing, microlensing, micro_contributions)

            # Display all observations:
            # visualise.plot_observations(time_series)

        # ____________________________________________________________________________


        # Save the desired quantities in the data frame
        df = write_to_df(df, index, batch_size, np.nan, z_source, z_lens, H_0, theta_E, obs_peak, obs_days,
                         obs_filters, model_mag, obs_mag, obs_mag_error, obs_snr, obs_mag_unresolved, mag_unresolved_error,
                         snr_unresolved, macro_mag, source_x, source_y,
                         td_images, time_delay_distance, x_image, y_image, gamma_lens, e1_lens, e2_lens, gamma1,
                         gamma2, micro_kappa, micro_gamma, micro_s, micro_peak, x1, c, M_B, obs_start, obs_end,
                         mult_method_peak, mult_method, mult_method_micro, mag_method_peak, mag_method, mag_method_micro,
                         coords, obs_skybrightness, obs_psf, obs_lim_mag, obs_N_coadds, survey, rolling, obs_mag_micro,
                         mag_micro_error, obs_snr_micro, mag_unresolved_micro, mag_unresolved_micro_error,
                         snr_unresolved_micro)

        # Check if the data frame is full
        if (index+1) % batch_size == 0 and index > 1:
            if Save:
                # Save data frame to laptop
                df.to_pickle(path + "Baselinev30_microlensing_numimages=" + str(int(num_images)) + "_batch" + str(str(batch).zfill(3)) + ".pkl")

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

    # print("Timing dict: ", timer.timing_dict)

    return df, timer.timing_dict


def main():
    # ---------------------------------
    telescope = 'LSST'
    bandpasses = ['r', 'i', 'z', 'y', 'g']
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

