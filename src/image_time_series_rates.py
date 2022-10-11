#! /bin/python3
import numpy as np
import sys
import time
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
from scipy.sparse import csr_matrix
from functions import create_dataframe, write_to_df, get_time_delay_distance
from class_lens import Lens
from class_supernova import Supernova
from class_microlensing import Microlensing
from class_visualisation import Visualisation
from class_LSST import Telescope

# from microlensing.create_db import *


def simulate_time_series_images(batch_size, batch, num_samples, num_images, obs_times, obs_filters, z_source_list_,
                                z_lens_list_, theta_E_list_, add_microlensing, telescope, bandpasses, obs_lower_limit,
                                obs_upper_limit, fixed_H0, Show, Save, path):

    """
    :param batch_size: number of lens systems that is saved together in a batch (int)
    :param num_samples: total number of lens systems to be generated (int)
    :param num_images: number of lensed supernova images. choose between 2 (for doubles) and 4 (for quads)
    :param obs_times: list containing observation times for a sample of supernovae according to LSST Baseline v2.0
    :param obs_filters: list containing filters for a sample of supernovae according to LSST Baseline v2.0
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

    lsst = Telescope(telescope, bandpasses)
    # Remove!
    bandpass = 'i'

    start_time = time.time()
    #start_t = time.time()
    tqdm._instances.clear()
    pbar = tqdm(total=num_samples)
    counter = 0                     # Total number of attempts
    attempts = 0                    # Counts number of attempts per configuration
    sample_index = 0                # Counts how many configurations have been used (including failed ones)
    index = 0                       # Counts how many successful configurations have been used

    timing1 = []
    timing2 = []
    timing3 = []
    timing4 = []
    timing5 = []
    timing6 = []
    timing7 = []
    timing8 = []
    timing9 = []
    timing10 = []
    timing11 = []
    mtiming1 = []
    mtiming2 = []

    mmtiming1, mmtiming2, mmtiming3, mmtiming4, mmtiming5, mmtiming6, mmtiming7 = [], [], [], [], [], [], []

    # =========================================== TIMING 1 ===========================================
    timing1_start = time.time()

    days_distribution = []

    if batch_size > num_samples:
        print("Error: batch_size cannot be larger than num_samples!")
        sys.exit()

    if num_images != 2 and num_images != 4:
        print("Error: num_images should be equal to 2 (for doubles) or 4 (for quads)")
        sys.exit()

    # Create Pandas dataframe to store the data
    df = create_dataframe(batch_size)

    # Sample num_samples from the joint z_lens, z_source, theta_E distribution
    # (Pick more samples since not all configurations will be successful)
    sample = np.random.choice(len(z_source_list_), size=6 * num_samples, replace=False)

    if add_microlensing:
        acceptance_macro = np.zeros(num_samples)
        acceptance_micro = np.zeros(num_samples)

    rejected_cadence_list = []
    accepted_peak_list = []
    rejected_cadence = 0
    accepted_peak = 0

    # _______________________________________________________________________

    timing1_end = time.time()
    timing1.append(timing1_end - timing1_start)


    while index < num_samples:

        # =========================================== TIMING 2 ===========================================
        timing2_start = time.time()
        # =========================================== TIMING 2 ===========================================

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
            H_0 = 70.0
        else:
            H_0 = np.random.uniform(20, 100)

        cosmo = FlatLambdaCDM(H0=H_0, Om0=0.3, Ob0=0.05)
        time_delay_distance = get_time_delay_distance(z_source, z_lens, cosmo)
        source_x = np.random.uniform(-theta_E, theta_E)
        source_y = np.random.uniform(-theta_E, theta_E)

        # =========================================== TIMING 2 ===========================================
        timing2_end = time.time()
        timing2.append(timing2_end - timing2_start)

        timing3_start = time.time()
        # =========================================== TIMING 3 ===========================================

        # Initiate the supernova and lens classes
        supernova = Supernova(theta_E, z_lens, z_source, cosmo, source_x, source_y)
        lens = Lens(theta_E, z_lens, z_source, cosmo)

        # _______________________________________________________________________

        # Lens specification
        lens_model_class, kwargs_lens, gamma_lens, e1_lens, e2_lens, gamma1, gamma2 = lens.mass_model()
        if np.isnan(gamma_lens):
            continue

        lens_light_model_class, kwargs_lens_light = lens.light_model()

        # Source specification (extended emission)
        source_model_class, kwargs_source = supernova.host_light_model()

        # Get image positions and magnifications
        x_image, y_image, macro_mag = supernova.get_image_pos_magnification(lens_model_class, kwargs_lens,
                                                                            min_distance=lsst.deltaPix,
                                                                            search_window=lsst.numPix * lsst.deltaPix)

        # =========================================== TIMING 3 ===========================================
        timing3_end = time.time()
        timing3.append(timing3_end - timing3_start)

        timing4_start = time.time()
        # =========================================== TIMING 4 ===========================================

        # _______________________________________________________________________

        # Checks

        # Is num_images equal to 2 for doubles and to 4 for quads?
        if len(x_image) != num_images:
            continue

        # Is maximum image separation between 0.5 and 4.0 arcsec?
        sep = supernova.separation(x_image, y_image)

        if sep < 0.5 or sep > 4.0:
            continue

        # _______________________________________________________________________

        # Time delays between images (geometric + gravitational)
        td_images = lens.time_delays(lens_model_class, kwargs_lens, x_image, y_image)

        # Supernova light curve
        model, x1, c, MW_dust, M_observed = supernova.light_curve(z_source)

        # Check peak brightness: detectable?
        if not supernova.check_detectability_peak(lsst, model, macro_mag, 0.0, False):
                continue

        if not add_microlensing:
            accepted_peak += 1

        # =========================================== TIMING 4  ===========================================
        timing4_end = time.time()
        timing4.append(timing4_end - timing4_start)

        timing5_start = time.time()
        # =========================================== TIMING 5 ===========================================

        # _______________________________________________________________________

        # Microlensing contributions

        if add_microlensing:
            # start_m = time.time()

            microlensing = Microlensing(lens_model_class, kwargs_lens, x_image, y_image, bandpass,
                                        theta_E, z_lens, z_source, cosmo)

            # =========================================== TIMING 5 ===========================================
            timing5_end = time.time()
            timing5.append(timing5_end - timing5_start)

            timing6_start = time.time()
            # =========================================== TIMING 6 ===========================================

            # Convergence
            micro_kappa = microlensing.get_kappa()

            # Shear
            micro_gamma = microlensing.get_gamma()

            # Smooth-matter fraction
            _, R_eff = lens.scaling_relations()
            micro_s = microlensing.get_s(R_eff)
            if np.any(np.isnan(micro_s)):
                continue

            # =========================================== TIMING 6 ===========================================
            timing6_end = time.time()
            timing6.append(timing6_end - timing6_start)

            timing7_start = time.time()
            # =========================================== TIMING 7 ===========================================

            # Load random microlensing light curves
            micro_lightcurves, macro_lightcurves, micro_times, mtiming1_, mtiming2_,\
                mmtiming1_, mmtiming2_, mmtiming3_, mmtiming4_, mmtiming5_, mmtiming6_, mmtiming7_ = microlensing.micro_lightcurve_all_images(micro_kappa,
                                                                                                         micro_gamma,
                                                                                                         micro_s)
            mtiming1.append(mtiming1_)
            mtiming2.append(mtiming2_)

            mmtiming1.append(mmtiming1_)
            mmtiming2.append(mmtiming2_)
            mmtiming3.append(mmtiming3_)
            mmtiming4.append(mmtiming4_)
            mmtiming5.append(mmtiming5_)
            mmtiming6.append(mmtiming6_)
            mmtiming7.append(mmtiming7_)

            # =========================================== TIMING 7 ===========================================
            timing7_end = time.time()
            timing7.append(timing7_end - timing7_start)

            timing8_start = time.time()
        # =========================================== TIMING 8 ===========================================


            # end_m = time.time()
            # print("Microlensing part: ", end_m - start_m, "seconds")


            """
            # ---------------------------------------
            # temporarily, for testing speed
            micro_lightcurves1, macro_lightcurves1, micro_times1 = get_microlensing('microlensing/microlensing_database.db',
                                                                                 kappa=0.956,
                                                                                 gamma=0.952,
                                                                                 s=0.616,
                                                                                 source_redshift=0.8,
                                                                                 bandpass='i')
            micro_lightcurves2, macro_lightcurves2, micro_times2 = get_microlensing(
                'microlensing/microlensing_database.db',
                kappa=0.956,
                gamma=0.952,
                s=0.616,
                source_redshift=0.8,
                bandpass='i')

            micro_lightcurves = np.array([micro_lightcurves1, micro_lightcurves2])
            macro_lightcurves = np.array([macro_lightcurves1, macro_lightcurves2])
            micro_times = np.array([micro_times1, micro_times2])
            # ---------------------------------------
            """

            # Calculate microlensing contribution at the peak
            micro_peak = microlensing.micro_snapshot(micro_lightcurves, macro_lightcurves, micro_times, td_images,
                                                     0, peak=True)

            # Check again if the peak brightness is detectable after microlensing
            if not supernova.check_detectability_peak(lsst, model, macro_mag, micro_peak, add_microlensing):
                continue

            # =========================================== TIMING 8 ===========================================
            timing8_end = time.time()
            timing8.append(timing8_end - timing8_start)


        else:
            micro_kappa = np.nan
            micro_gamma = np.nan
            micro_s = np.nan
            micro_peak = 0.0

        timing9_start = time.time()
        # =========================================== TIMING 9 ===========================================

        # _______________________________________________________________________

        # Generate image time series

        obs_index = int(np.random.uniform(0, len(obs_times)))

        time_series = []
        offset = np.random.uniform(0, max(obs_times[index]))
        start_obs = model.mintime() + min(td_images) + offset
        end_obs = model.maxtime() + max(td_images) + offset
        days = obs_times[obs_index][obs_times[obs_index] > start_obs]
        obs_days = []
        obs_days_filters = []

        # Keep track of the brightness of each observation
        brightness_obs = np.ones((obs_upper_limit, len(x_image))) * 50

        if add_microlensing:
            brightness_obs_micro = np.ones((obs_upper_limit, len(x_image))) * 50

        for observation in range(obs_upper_limit):

            if observation > len(days) + -1:
                break

            day = days[observation]
            filter = obs_filters[obs_index][observation]

            # For the r-filter, light curves with z > 1.5 are not defined. Skip these.
            if filter == 'r' and z_source > 1.5:
                continue

            if day > end_obs:
                break

            obs_days.append(day)
            obs_days_filters.append(filter)

            # Calculate microlensing contribution to light curve on this specific point in time
            if add_microlensing:
                micro_day = microlensing.micro_snapshot(micro_lightcurves, macro_lightcurves, micro_times,
                                                        td_images, day)
            else:
                micro_day = np.nan

            # Calculate apparent magnitudes without microlensing
            app_mag_ps = supernova.get_app_magnitude(model, day, macro_mag, td_images, np.nan, telescope, filter, False)
            brightness_obs[observation] = app_mag_ps

            if add_microlensing:
                # Calculate brightness with microlensing
                app_mag_ps_micro = supernova.get_app_magnitude(model, day, macro_mag, td_images, micro_day, telescope,
                                                               filter, add_microlensing)
                brightness_obs_micro[observation] = app_mag_ps_micro


            # Calculate amplitude parameter
            amp_ps = lsst.app_mag_to_amplitude(app_mag_ps, filter)

            # Create the image and save it to the time-series list
            image_sim = lsst.generate_image(x_image, y_image, amp_ps, lens_model_class, source_model_class,
                                            lens_light_model_class, kwargs_lens, kwargs_source, kwargs_lens_light, filter)

            time_series.append(image_sim)

        # =========================================== TIMING 9 ===========================================
        timing9_end = time.time()
        timing9.append(timing9_end - timing9_start)

        timing10_start = time.time()
        # =========================================== TIMING 10 ===========================================

        # _______________________________________________________________________

        # Final cuts

        # Determine whether the lensed SN is detectable, based on its brightness and flux ratio

        # Without microlensing:
        if supernova.check_detectability(lsst, model, macro_mag, brightness_obs, obs_days_filters, 0.0, False):
            acceptance_macro[index] += 1

        if add_microlensing:
            # With microlensing:
            if supernova.check_detectability(lsst, model, macro_mag, brightness_obs, obs_days_filters, micro_peak,
                                             add_microlensing):
                acceptance_micro[index] += 1


        """
        Probably not necessary anymore?

        # Count the acceptance fraction due to inter-night gaps
        # How many objects are rejected due to their brightness, after passing the peak-brightness cut?
        # NOTE: for microlensing, new peak-brightness should be determined!

        # If microlensing: determine the new peak brightness, and whether this passes the cut
        if add_microlensing:
            peak_brightness_micro = peak_brightness + micro_peak

            if num_images == 2:
                if len(peak_brightness_micro[peak_brightness_micro < lsst.limiting_magnitude]) < 2:
                    continue
            elif num_images == 4:
                if len(peak_brightness_micro[peak_brightness_micro < lsst.limiting_magnitude]) < 3:
                    continue

            accepted_peak += 1

        # Now look at the brightest image: does it pass the cut?
        # For doubles: 2 images/for quads: 3 images should be below (brighter than) the LSST limiting magnitude
        if num_images == 2:
            if len(peak_brightness_image[peak_brightness_image < lsst.limiting_magnitude]) < 2:
                rejected_cadence += 1
                continue
        elif num_images == 4:
            if len(peak_brightness_image[peak_brightness_image < lsst.limiting_magnitude]) < 3:
                rejected_cadence += 1
                continue

        # For doubles: check if flux ratio is larger than 0.1
        flux_ratio = supernova.flux_ratio(model, macro_mag, micro_peak, add_microlensing)
        if flux_ratio <= 0.1 or flux_ratio >= 10:
            continue
        """

        # Discard systems with fewer than obs_lower_limit images
        L = len(time_series)
        if L < obs_lower_limit:
            continue

        # Cut out anything above obs_upper_limit observations
        if L > obs_upper_limit:
            del time_series[obs_upper_limit:]

        # Fill up time_series < obs_upper_limit with zero padding
        if L < obs_upper_limit:

            filler = np.zeros((48, 48))

            for i in range(obs_upper_limit - L):
                time_series.append(csr_matrix(filler))

        # =========================================== TIMING 10 ===========================================
        timing10_end = time.time()
        timing10.append(timing10_end - timing10_start)

        timing11_start = time.time()
        # =========================================== TIMING 11 ===========================================


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

            visualise = Visualisation(time_delay_distance, td_images, theta_E, data_class_i, macro_mag, obs_days, obs_days_filters)

            # Print the properties of the lensed supernova system
            visualise.print_properties(z_lens, z_source, H_0, micro_peak)

            # Plot time delay surface
            visualise.plot_td_surface(lens_model_class, kwargs_lens, source_x, source_y, x_image, y_image)

            # Plot light curve with observation epochs
            visualise.plot_light_curves(model, supernova.bandpass, day_range, micro_day_range, add_microlensing)

            # Display all observations:
            visualise.plot_observations(time_series)

        # ____________________________________________________________________________

        if add_microlensing:
            acceptance_fraction = acceptance_micro[index]/acceptance_macro[index]
        else:
            acceptance_fraction = 1.0

        # Save the desired quantities in the data frame
        df = write_to_df(df, index, batch_size, time_series, z_source, z_lens, H_0, theta_E, peak_brightness_image,
                         macro_mag, source_x, source_y, td_images, time_delay_distance, x_image, y_image, gamma_lens,
                         e1_lens, e2_lens, days, gamma1, gamma2, micro_kappa, micro_gamma, micro_s, micro_peak,
                         acceptance_fraction)

        # Check if the data frame is full
        if (index+1) % batch_size == 0 and index > 1:
            if Save:
                # Save data frame to laptop
                df.to_pickle(path + "LSST_numimages=" + str(int(num_images)) + "_fixedH0=" +
                str(fixed_H0) + "_microlensing=" + str(add_microlensing) + "_batch" + str(str(batch).zfill(3)) + ".pkl")

            if (index+1) < num_samples:
                # Start a new, empty data frame
                df = create_dataframe(batch_size)
            batch += 1

        days_distribution.append(len(days))
        rejected_cadence_list.append(rejected_cadence)
        accepted_peak_list.append(accepted_peak)

        # Update variables
        sample_index += 1
        index += 1
        pbar.update(1)
        attempts = 0
        rejected_cadence = 0
        accepted_peak = 0

        # =========================================== TIMING 11 ===========================================
        timing11_end = time.time()
        timing11.append(timing11_end - timing11_start)


    end_time = time.time()
    duration = end_time - start_time

    print("Timing results")
    print("Len: ", len(timing1), len(timing2), len(timing3), len(timing4), len(timing5), len(timing6), len(timing7),
          len(timing8), len(timing9), len(timing10), len(timing11))
    print(" ")

    print("Mean time: ", np.mean(timing1), np.mean(timing2), np.mean(timing3), np.mean(timing4), np.mean(timing5),
          np.mean(timing6), np.mean(timing7), np.mean(timing8), np.mean(timing9), np.mean(timing10), np.mean(timing11))
    print(" ")
    #end_t = time.time()
    #print("Total: ", end_t - start_t, "seconds")

    if add_microlensing and Save:
        acceptance_fractions = np.array(acceptance_micro) / np.array(acceptance_macro)
        np.savetxt(path + "data/Acceptance_fractions_microlensing.txt", acceptance_fractions)

    acceptance_fraction_cadence = (np.array(accepted_peak_list) - np.array(rejected_cadence_list)) / np.array(accepted_peak_list)
    if Save:
        np.savetxt(path + "data/Acceptance_fractions_cadence.txt", acceptance_fraction_cadence)
        np.savetxt(path + "data/max_obs_distribution.txt", days_distribution)

    print("Done!")
    print("Simulating images took ", np.around(duration), "seconds (", np.around(duration / 3600, 2), "hours) to complete.")
    print("Number of image-time series generated: ", index)
    print("Number of configurations tried: ", sample_index)
    print("Number of attempts: ", counter)
    print(" ")
    print(df)

    return [timing1, timing2, timing3, timing4, timing5, timing6, timing7, timing8, timing9, timing10, timing11,
            mtiming1, mtiming2], [mmtiming1, mmtiming2, mmtiming3, mmtiming4, mmtiming5, mmtiming6, mmtiming7]


def main():

    print("test")


if __name__ == '__main__':
    main()

