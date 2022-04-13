#! /bin/python3
import numpy as np
import pandas as pd
import math


def create_dataframe(batch_size):
    """
    Creates an empty pandas data frame, to be filled with information from each run of the simulation.

    :param batch_size: number of rows (corresponding to lens systems) the data frame should contain
    :return: an empty pandas data frame with [batch_size] rows and 23 columns
    """
    df = pd.DataFrame(np.zeros((batch_size, 24)),
         columns=['time_series', 'z_source', 'z_lens', 'H0', 'theta_E',
         'peak_brightness_im', 'macro_mag', 'source_x', 'source_y', 'time_delay',
         'time_delay_distance', 'image_x', 'image_y', 'gamma_lens',
          'e1_lens', 'e2_lens', 'time_stamps', 'g1_shear', 'g2_shear', 'micro_kappa',
                  'micro_gamma', 'micro_s', 'micro_peak', 'acceptance_fraction'])

    df['time_series'] = df['time_series'].astype('object')
    df['time_delay'] = df['time_delay'].astype('object')
    df['macro_mag'] = df['macro_mag'].astype('object')
    df['image_x'] = df['image_x'].astype('object')
    df['image_y'] = df['image_y'].astype('object')
    df['peak_brightness_im'] = df['peak_brightness_im'].astype('object')
    df['time_stamps'] = df['time_stamps'].astype('object')
    df['micro_kappa'] = df['micro_kappa'].astype('object')
    df['micro_gamma'] = df['micro_gamma'].astype('object')
    df['micro_s'] = df['micro_s'].astype('object')
    df['micro_peak'] = df['micro_peak'].astype('object')

    return df


def write_to_df(df, index, batch_size, time_series, z_source, z_lens, H_0, theta_E, peak_brightness_image, macro_mag,
                source_x, source_y, td_images, time_delay_distance, x_image, y_image, gamma_lens, e1_lens, e2_lens,
                days, gamma1, gamma2, micro_kappa, micro_gamma, micro_s, micro_peak, acceptance_fraction):
    """
    Write the properties of the current lens system into a row of the data frame.

    :param df: pandas data frame of size [batch_size x 18] containing the properties of the saved lens systems
    :param index: count of how many successful configurations have been used to generate lens systems
    :param batch_size: number of rows (corresponding to lens systems) the data frame should contain
    :param time_series: list of length [obs_upper_limit] containing simulated images of the lensed supernova at
           different time stamps corresponding to different observations
    :param z_source: redshift of the supernova (float)
    :param z_lens: redshift of the lens galaxy (float)
    :param H_0: value of the Hubble constant used for the background cosmology in this current lens configuration (float)
    :param theta_E: einstein radius of the lens galaxy (float)
    :param peak_brightness_image: array of length [num_images] containing the brightest apparent magnitudes of each of
           the supernova images
    :param macro_mag: array of length [num_images] with the macro magnification for each image
    :param source_x: x-position of the supernova relative to the lens galaxy in arcsec (float)
    :param source_y: y-position of the supernova relative to the lens galaxy in arcsec (float)
    :param td_images: array of length [num_images] containing the relative time delays between the supernova images
    :param time_delay_distance: time delay distance of the lens system in Mpc (float)
    :param x_image: array of length [num_images] containing the x coordinates of the supernova images in arcsec
    :param y_image: array of length [num_images] containing the y coordinates of the supernova images in arcsec
    :param gamma_lens: power-law slope of the lens mass profile (float)
    :param e1_lens: ellipticity component of the lens mass profile (float)
    :param e2_lens: ellipticity component of the lens mass profile (float)
    :param days: array containing the time stamps (in days) of the observations
    :param gamma1: component of external shear (float)
    :param gamma2: component of external shear (float)
    :param micro_kappa: array of length [num_images] containing the convergence/kappa for each image
    :param micro_gamma: array of length [num_images] containing smooth matter fraction for each image
    :param micro_peak: array of length [num_images] containing the microlensing contributions at light curve peak
    :param acceptance_fraction: number of accepted microlensing realisations / number of accepted macrolensing realisations
    :return: pandas data frame of size [batch_size x 18] containing the properties of the saved lens systems,
             including the newest one
    """
    df['time_series'][index % batch_size] = time_series
    df['z_source'][index % batch_size] = z_source
    df['z_lens'][index % batch_size] = z_lens
    df['H0'][index % batch_size] = H_0
    df['theta_E'][index % batch_size] = theta_E
    df['peak_brightness_im'][index % batch_size] = peak_brightness_image
    df['macro_mag'][index % batch_size] = macro_mag
    df['source_x'][index % batch_size] = source_x
    df['source_y'][index % batch_size] = source_y
    df['time_delay'][index % batch_size] = td_images
    df['time_delay_distance'][index % batch_size] = time_delay_distance
    df['image_x'][index % batch_size] = x_image
    df['image_y'][index % batch_size] = y_image
    df['gamma_lens'][index % batch_size] = gamma_lens
    df['e1_lens'][index % batch_size] = e1_lens
    df['e2_lens'][index % batch_size] = e2_lens
    df['time_stamps'][index % batch_size] = days - min(days)
    df['g1_shear'][index % batch_size] = gamma1
    df['g2_shear'][index % batch_size] = gamma2
    df['micro_kappa'][index % batch_size] = micro_kappa
    df['micro_gamma'][index % batch_size] = micro_gamma
    df['micro_s'][index % batch_size] = micro_s
    df['micro_peak'][index % batch_size] = micro_peak
    df['acceptance_fraction'][index % batch_size] = acceptance_fraction
    return df


def get_time_delay_distance(z_source, z_lens, cosmo):
    """
    Calculate the time delay distance from the lens and source redshifts and cosmology.

    :param z_source: redshift of the supernova (float)
    :param z_lens: redshift of the lens galaxy (float)
    :param cosmo: instance of astropy containing the background cosmology
    :return: time delay distance of the lens system in Mpc (float)
    """
    D_dt = (1 + z_lens) * cosmo.angular_diameter_distance(z_lens).value * \
            cosmo.angular_diameter_distance(z_source).value / \
            cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value
    return D_dt


def find_nearest(array, value):
    """
    Selects the value from an array that is closest to a certain input number.

    :param array: the array from which a value needs to be returned
    :param value: the output array value should be closest to this number
    :return: the value from array that is closest to the input number
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1], idx-1
    else:
        return array[idx], idx


def main():

    print("test")


if __name__ == '__main__':
    main()

