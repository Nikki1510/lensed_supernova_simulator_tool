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

# from microlensing.create_db import *


def simulate_time_series_images(batch_size, batch, num_samples, num_images, inter_night_gap, z_source_list_, z_lens_list_,
                                theta_E_list_, lsst, bandpass, add_microlensing, obs_lower_limit, obs_upper_limit,
                                fixed_H0, Show, Save, path):
    """

    :param batch_size: number of lens systems that is saved together in a batch (int)
    :param num_samples: total number of lens systems to be generated (int)
    :param num_images: number of lensed supernova images. choose between 2 (for doubles) and 4 (for quads)
    :param inter_night_gap: 1D array containing the distribution of LSST inter night gaps between observations
    :param z_source_list_: array containing ~ 400,000 values of the source redshift
    :param z_lens_list_: array containing ~ 400,000 values of the lens redshift
    :param theta_E_list_: array containing ~ 400,000 values of the einstein radius
    :param lsst:
    :param bandpass:
    :param add_microlensing:
    :param obs_lower_limit:
    :param obs_upper_limit:
    :param fixed_H0:
    :param Show:
    :param Save:
    :param path:
    :return: Generates image time-series and saves them to a pickle file
    """
    start_time = time.time()
    #start_t = time.time()
    tqdm._instances.clear()
    pbar = tqdm(total=num_samples)
    counter = 0                     # Total number of attempts
    # batch = 1                       # Number associated with the first batch
    attempts = 0                    # Counts number of attempts per configuration
    sample_index = 0                # Counts how many configurations have been used (including failed ones)
    index = 0                       # Counts how many successful configurations have been used


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

    while index < num_samples:
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

        # Initiate the supernova and lens classes
        supernova = Supernova(theta_E, z_lens, z_source, cosmo, source_x, source_y, bandpass)
        lens = Lens(theta_E, z_lens, z_source, cosmo, bandpass)

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
        peak_brightness = model.bandmag(supernova.bandpass, time=0, magsys='ab')
        peak_brightness -= 2.5 * np.log10(macro_mag)

        if num_images == 2:
            if len(peak_brightness[peak_brightness < lsst.limiting_magnitude]) < 2:
                continue
        elif num_images == 4:
            if len(peak_brightness[peak_brightness < lsst.limiting_magnitude]) < 3:
                continue

        if not add_microlensing:
            accepted_peak += 1

        # _______________________________________________________________________

        # Microlensing contributions

        if add_microlensing:
            # start_m = time.time()

            microlensing = Microlensing(lens_model_class, kwargs_lens, x_image, y_image, bandpass,
                                        theta_E, z_lens, z_source, cosmo)

            # Convergence
            micro_kappa = microlensing.get_kappa()

            # Shear
            micro_gamma = microlensing.get_gamma()

            # Smooth-matter fraction
            _, R_eff = lens.scaling_relations()
            micro_s = microlensing.get_s(R_eff)
            if np.any(np.isnan(micro_s)):
                continue

            # Load random microlensing light curves
            micro_lightcurves, macro_lightcurves, micro_times = microlensing.micro_lightcurve_all_images(micro_kappa,
                                                                                                         micro_gamma,
                                                                                                         micro_s)

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


        else:
            micro_kappa = np.nan
            micro_gamma = np.nan
            micro_s = np.nan
            micro_peak = 0.0

        # _______________________________________________________________________

        # Generate image time series

        sigma_bkg = lsst.background_noise()
        data_class, x_grid1d, y_grid1d, min_coordinate, max_coordinate = lsst.grid(sigma_bkg)

        time_series = []
        days = []
        day = model.mintime() + min(td_images) + 10
        max_day = model.maxtime() + max(td_images)

        peak_brightness_image = np.ones(len(x_image)) * 50
        if add_microlensing:
            peak_brightness_image_macro = np.ones(len(x_image)) * 50

        for observation in range(obs_upper_limit):

            if day > max_day:
                break

            days.append(day)

            # Calculate microlensing contribution to light curve on this specific point in time
            if add_microlensing:
                micro_day = microlensing.micro_snapshot(micro_lightcurves, macro_lightcurves, micro_times,
                                                        td_images, day)
            else:
                micro_day = np.nan

            # Calculate apparent magnitudes
            app_mag_ps, peak_brightness_image = supernova.get_app_magnitude(model, day, macro_mag, td_images,
                                                                            peak_brightness_image, micro_day,
                                                                            add_microlensing)

            if add_microlensing:
                # Calculate brightness WITHOUT microlensing
                app_mag_ps_macro, peak_brightness_image_macro = supernova.get_app_magnitude(model, day, macro_mag,
                                                                td_images, peak_brightness_image_macro, np.nan, False)

            # Calculate amplitude parameter
            amp_ps = lsst.app_mag_to_amplitude(app_mag_ps)

            # Create the image and save it to the time-series list
            image_sim = lsst.generate_image(x_image, y_image, amp_ps, data_class, lens_model_class, source_model_class,
                                            lens_light_model_class, kwargs_lens, kwargs_source, kwargs_lens_light,
                                            sigma_bkg)

            time_series.append(image_sim)

            # Determine next observation time
            day += np.random.choice(inter_night_gap)

        # _______________________________________________________________________

        # Final cuts

        # Count the acceptance fraction due to microlensing effects (on brightness & flux ratio)
        if add_microlensing:
            if num_images == 2:
                flux_ratio = supernova.flux_ratio(model, macro_mag, micro_peak, add_microlensing)
                flux_ratio_macro = supernova.flux_ratio(model, macro_mag, np.nan, False)
                # Check if the macro magnified images pass the brightness cut
                if len(peak_brightness_image_macro[peak_brightness_image_macro < lsst.limiting_magnitude]) == 2 and \
                    flux_ratio_macro > 0.1 and flux_ratio_macro < 10:
                    acceptance_macro[index] += 1

                # Check if the micro magnified images pass the brightness cut
                if len(peak_brightness_image[peak_brightness_image < lsst.limiting_magnitude]) == 2 and \
                        flux_ratio > 0.1 and flux_ratio < 10:
                    acceptance_micro[index] += 1

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

            visualise = Visualisation(time_delay_distance, td_images, theta_E, data_class, macro_mag, days)

            # Print the properties of the lensed supernova system
            visualise.print_properties(peak_brightness_image, z_lens, z_source, H_0, micro_peak)

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
                df.to_pickle(path + "LSST_numimages=" + str(int(num_images)) + "_band=" + str(bandpass) + "_fixedH0=" +
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

    end_time = time.time()
    duration = end_time - start_time

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


def main():

    print("test")


if __name__ == '__main__':
    main()

