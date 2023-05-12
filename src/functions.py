#! /bin/python3
import numpy as np
import pandas as pd
import math


def create_dataframe_unlensed(batch_size):
    """
    Creates an empty pandas data frame, to be filled with information from each run of the simulation.

    :param batch_size: number of rows (corresponding to lens systems) the data frame should contain
    :return: an empty pandas data frame with [batch_size] rows and 23 columns
    """
    df = pd.DataFrame(np.zeros((batch_size, 21)),
         columns=['z_source', 'H0', 'obs_peak', 'obs_times', 'obs_bands', 'model_mag', 'obs_mag', 'obs_mag_error',
                  'obs_snr', 'stretch', 'colour', 'Mb', 'obs_start', 'obs_end',
                  'coords', 'obs_skybrightness', 'obs_psf', 'obs_lim_mag', 'obs_N_coadds', 'survey', 'rolling'])

    df['obs_peak'] = df['obs_peak'].astype('object')
    df['obs_times'] = df['obs_times'].astype('object')
    df['obs_bands'] = df['obs_bands'].astype('object')
    df['model_mag'] = df['model_mag'].astype('object')
    df['obs_mag'] = df['obs_mag'].astype('object')
    df['obs_mag_error'] = df['obs_mag_error'].astype('object')
    df['obs_snr'] = df['obs_snr'].astype('object')
    df['coords'] = df['coords'].astype('object')
    df['obs_skybrightness'] = df['obs_skybrightness'].astype('object')
    df['obs_psf'] = df['obs_psf'].astype('object')
    df['obs_lim_mag'] = df['obs_lim_mag'].astype('object')
    df['obs_N_coadds'] = df['obs_N_coadds'].astype('object')

    return df


def write_to_df_unlensed(df, index, batch_size, z_source, H_0, obs_peak, obs_times, obs_bands, model_mag,
                         obs_mag, obs_mag_error, obs_snr, stretch, colour, Mb, obs_start, obs_end,
                         coords, obs_skybrightness, obs_psf, obs_lim_mag, obs_N_coadds, survey, rolling):
    """
    Write the properties of the current lens system into a row of the data frame.

    :param df: pandas data frame of size [batch_size x 18] containing the properties of the saved lens systems
    :param index: count of how many successful configurations have been used to generate lens systems
    :param batch_size: number of rows (corresponding to lens systems) the data frame should contain
    :param z_source: redshift of the supernova (float)
    :param H_0: value of the Hubble constant used for the background cosmology in this current lens configuration (float)
    :param obs_peak: array of length [num_filters, num_images] containing the brightest apparent magnitudes of each of
           the supernova images for each filter/bandpass
    :param obs_times: array of length N_observations containing the time stamps of each observation
    :param obs_bands: array of length N_observations containing the bandpasses for each observation
    :param model_mag: array of shape [N_observations, N_images] that contains the apparent magnitudes for each
            observation and each image as predicted by the model (so NOT perterbed by weather)
    :param obs_mag: array of shape [N_observations, N_images] that contains the apparent magnitudes for each
            observation and each image (perterbed by weather). Use in combination with obs_times and obs_bands.
    :param obs_mag_error: array of shape [N_observations, N_images] containing the errors on the apparent magnitudes
    :param obs_snr: array of shape [N_observations, N_images] containing the S/N ratio for each image
    :param stretch: stretch parameter associated with the supernova light curve (float)
    :param colour: colour parameter associated with the supernova light curve (float)
    :param Mb: absolute magnitude of the unlensed supernova in the B band (float)
    :param obs_start: MJD of the first observation (float)
    :param obs_end: MJD of the last observation (float)
    :param coords: right ascension and declination of the lensed supernova
    :param obs_skybrightness: array of length N_observations containing the sky brightness (in magnitudes)
    :param obs_psf: array of length N_observations containing the FWHM of the PSF for each observation (in arcsec)
    :param obs_lim_mag: array of length N_observations containing the limiting magnitude (5 sigma depth)
    :param obs_N_coadds: array of length N_observations with the number of coadds for each observation
    :param survey: whether the sky coordinates belong to the WFD/DDF/galactic plane and pole region
    :param rolling: only for WFD; whether the cadence does not rol or is in the active or background rolling region.
    :return: pandas data frame of size [batch_size x 18] containing the properties of the saved lens systems,
             including the newest one
    """

    df['z_source'][index % batch_size] = z_source
    df['H0'][index % batch_size] = H_0
    df['obs_peak'][index % batch_size] = obs_peak
    df['obs_times'][index % batch_size] = obs_times
    df['obs_bands'][index % batch_size] = obs_bands
    df['model_mag'][index % batch_size] = model_mag
    df['obs_mag'][index % batch_size] = obs_mag
    df['obs_mag_error'][index % batch_size] = obs_mag_error
    df['obs_snr'][index % batch_size] = obs_snr
    df['stretch'][index % batch_size] = stretch
    df['colour'][index % batch_size] = colour
    df['Mb'][index % batch_size] = Mb
    df['obs_start'][index % batch_size] = obs_start
    df['obs_end'][index % batch_size] = obs_end
    df['coords'][index % batch_size] = coords
    df['obs_skybrightness'][index % batch_size] = obs_skybrightness
    df['obs_psf'][index % batch_size] = obs_psf
    df['obs_lim_mag'][index % batch_size] = obs_lim_mag
    df['obs_N_coadds'][index % batch_size] = obs_N_coadds
    df['survey'][index % batch_size] = survey
    df['rolling'][index % batch_size] = rolling
    return df


def create_dataframe(batch_size):
    """
    Creates an empty pandas data frame, to be filled with information from each run of the simulation.

    :param batch_size: number of rows (corresponding to lens systems) the data frame should contain
    :return: an empty pandas data frame with [batch_size] rows and 23 columns
    """
    df = pd.DataFrame(np.zeros((batch_size, 55)),
         columns=['time_series', 'z_source', 'z_lens', 'H0', 'theta_E', 'obs_peak', 'obs_times', 'obs_bands', 'model_mag',
                  'obs_mag', 'obs_mag_error', 'obs_snr', 'obs_mag_unresolved', 'mag_unresolved_error', 'snr_unresolved',
                  'macro_mag', 'source_x', 'source_y', 'time_delay', 'time_delay_distance', 'image_x', 'image_y',
                  'gamma_lens', 'e1_lens', 'e2_lens', 'g1_shear', 'g2_shear', 'micro_kappa', 'micro_gamma', 'micro_s',
                  'micro_peak', 'stretch', 'colour', 'Mb', 'obs_start', 'obs_end', 'mult_method_peak', 'mult_method',
                  'mult_method_micro', 'mag_method_peak', 'mag_method', 'mag_method_micro',  'coords',
                  'obs_skybrightness', 'obs_psf', 'obs_lim_mag', 'obs_N_coadds', 'survey', 'rolling', 'obs_mag_micro',
                  'mag_micro_error', 'obs_snr_micro', 'mag_unresolved_micro', 'mag_unresolved_micro_error',
                  'snr_unresolved_micro'])

    df['time_series'] = df['time_series'].astype('object')
    df['time_delay'] = df['time_delay'].astype('object')
    df['obs_peak'] = df['obs_peak'].astype('object')
    df['obs_times'] = df['obs_times'].astype('object')
    df['obs_bands'] = df['obs_bands'].astype('object')
    df['model_mag'] = df['model_mag'].astype('object')
    df['obs_mag'] = df['obs_mag'].astype('object')
    df['obs_mag_error'] = df['obs_mag_error'].astype('object')
    df['obs_snr'] = df['obs_snr'].astype('object')
    df['obs_mag_unresolved'] = df['obs_mag_unresolved'].astype('object')
    df['mag_unresolved_error'] = df['mag_unresolved_error'].astype('object')
    df['snr_unresolved'] = df['snr_unresolved'].astype('object')
    df['macro_mag'] = df['macro_mag'].astype('object')
    df['image_x'] = df['image_x'].astype('object')
    df['image_y'] = df['image_y'].astype('object')
    df['micro_kappa'] = df['micro_kappa'].astype('object')
    df['micro_gamma'] = df['micro_gamma'].astype('object')
    df['micro_s'] = df['micro_s'].astype('object')
    df['micro_peak'] = df['micro_peak'].astype('object')
    df['coords'] = df['coords'].astype('object')
    df['obs_skybrightness'] = df['obs_skybrightness'].astype('object')
    df['obs_psf'] = df['obs_psf'].astype('object')
    df['obs_lim_mag'] = df['obs_lim_mag'].astype('object')
    df['obs_N_coadds'] = df['obs_N_coadds'].astype('object')

    df['obs_mag_micro'] = df['obs_mag_micro'].astype('object')
    df['mag_micro_error'] = df['mag_micro_error'].astype('object')
    df['obs_snr_micro'] = df['obs_snr_micro'].astype('object')
    df['mag_unresolved_micro'] = df['mag_unresolved_micro'].astype('object')
    df['mag_unresolved_micro_error'] = df['mag_unresolved_micro_error'].astype('object')
    df['snr_unresolved_micro'] = df['snr_unresolved_micro'].astype('object')

    return df


def write_to_df(df, index, batch_size, time_series, z_source, z_lens, H_0, theta_E, obs_peak, obs_times, obs_bands,
                model_mag, obs_mag, obs_mag_error, obs_snr, obs_mag_unresolved, mag_unresolved_error, snr_unresolved,
                macro_mag, source_x, source_y, td_images, time_delay_distance, x_image, y_image,
                gamma_lens, e1_lens, e2_lens, gamma1, gamma2, micro_kappa, micro_gamma, micro_s, micro_peak,
                stretch, colour, Mb, obs_start, obs_end, mult_method_peak, mult_method, mult_method_micro,
                mag_method_peak, mag_method, mag_method_micro, coords, obs_skybrightness, obs_psf, obs_lim_mag,
                obs_N_coadds, survey, rolling, obs_mag_micro, mag_micro_error, obs_snr_micro, mag_unresolved_micro,
                mag_unresolved_micro_error, snr_unresolved_micro):
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
    :param obs_peak: array of length [num_filters, num_images] containing the brightest apparent magnitudes of each of
           the supernova images for each filter/bandpass
    :param obs_times: array of length N_observations containing the time stamps of each observation
    :param obs_bands: array of length N_observations containing the bandpasses for each observation
    :param model_mag: array of shape [N_observations, N_images] that contains the apparent magnitudes for each
            observation and each image as predicted by the model (so NOT perterbed by weather)
    :param obs_mag: array of shape [N_observations, N_images] that contains the apparent magnitudes for each
            observation and each image (perterbed by weather). Use in combination with obs_times and obs_bands.
    :param obs_mag_error: array of shape [N_observations, N_images] containing the errors on the apparent magnitudes
    :param obs_snr: array of shape [N_observations, N_images] containing the S/N ratio for each image
    :param obs_mag_unresolved: array of len N_observations, containing the apparent magnitude for all SN images together,
            corresponding to the case of unresolved images
    :param mag_unresolved_error: array of len N_observations, containing the magnitude errors for the unresolved magnitudes
    :param snr_unresolved: array of len N_observations, containing the S/N ratio for the unresolved observations
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
    :param gamma1: component of external shear (float)
    :param gamma2: component of external shear (float)
    :param micro_kappa: array of length [num_images] containing the convergence/kappa for each image
    :param micro_gamma: array of length [num_images] containing the shear for each image
    :param micro_s: array of length [num_images] containing smooth matter fraction for each image
    :param micro_peak: array of length [num_images] containing the microlensing contributions at light curve peak
    :param stretch: stretch parameter associated with the supernova light curve (float)
    :param colour: colour parameter associated with the supernova light curve (float)
    :param Mb: absolute magnitude of the unlensed supernova in the B band (float)
    :param obs_start: MJD of the first observation (float)
    :param obs_end: MJD of the last observation (float)
    :param mult_method_peak: Bool. if True: detected with the multiplicity method at peak (corresponds to detection
            numbers in Wojtak et al.)
    :param mult_method: Bool. if True: detected with multiplicity method in actual observations
    :param mult_method_micro: Bool. if True: detected with multiplicity method when microlensing is included
    :param mag_method_peak: Bool. if True: detected with the magnification method at peak (corresponds to detection
            numbers in Wojtak et al.)
    :param mag_method: Bool. if True: detected with magnification method in actual observations
    :param mag_method_micro: Bool. if True: detected with magnification method when microlensing is included
    :param coords: right ascension and declination of the lensed supernova
    :param obs_skybrightness: array of length N_observations containing the sky brightness (in magnitudes)
    :param obs_psf: array of length N_observations containing the FWHM of the PSF for each observation (in arcsec)
    :param obs_lim_mag: array of length N_observations containing the limiting magnitude (5 sigma depth)
    :param obs_N_coadds: array of length N_observations with the number of coadds for each observation
    :param survey: whether the sky coordinates belong to the WFD/DDF/galactic plane and pole region
    :param rolling: only for WFD; whether the cadence does not rol or is in the active or background rolling region.
    :param obs_mag_micro: array of shape [N_observations, N_images] that contains the apparent magnitudes including
           microlensing magnifications and perturbations due to weather
    :param mag_micro_error: array of shape [N_observations, N_images] containing the errors on the apparent magnitudes
           for the microlensed light curves
    :param obs_snr_micro: array of shape [N_observations, N_images] containing the S/N ratio for the microlensed curves
    :param mag_unresolved_micro: array of len N_observations, containing the unresolved magnitudes for microlensing
    :param mag_unresolved_micro_error: array of len N_observations, containing the magnitude errors for the unresolved
           microlensed light curves
    :param snr_unresolved_micro: array of len N_observations, containing the S/N of the unresolved microlensed curves

    :return: pandas data frame of size [batch_size x 18] containing the properties of the saved lens systems,
             including the newest one
    """
    df['time_series'][index % batch_size] = time_series
    df['z_source'][index % batch_size] = z_source
    df['z_lens'][index % batch_size] = z_lens
    df['H0'][index % batch_size] = H_0
    df['theta_E'][index % batch_size] = theta_E
    df['obs_peak'][index % batch_size] = obs_peak
    df['obs_times'][index % batch_size] = obs_times
    df['obs_bands'][index % batch_size] = obs_bands
    df['model_mag'][index % batch_size] = model_mag
    df['obs_mag'][index % batch_size] = obs_mag
    df['obs_mag_error'][index % batch_size] = obs_mag_error
    df['obs_snr'][index % batch_size] = obs_snr
    df['obs_mag_unresolved'][index % batch_size] = obs_mag_unresolved
    df['mag_unresolved_error'][index % batch_size] = mag_unresolved_error
    df['snr_unresolved'][index % batch_size] = snr_unresolved
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
    df['g1_shear'][index % batch_size] = gamma1
    df['g2_shear'][index % batch_size] = gamma2
    df['micro_kappa'][index % batch_size] = micro_kappa
    df['micro_gamma'][index % batch_size] = micro_gamma
    df['micro_s'][index % batch_size] = micro_s
    df['micro_peak'][index % batch_size] = micro_peak
    df['stretch'][index % batch_size] = stretch
    df['colour'][index % batch_size] = colour
    df['Mb'][index % batch_size] = Mb
    df['obs_start'][index % batch_size] = obs_start
    df['obs_end'][index % batch_size] = obs_end
    df['mult_method_peak'][index % batch_size] = mult_method_peak
    df['mult_method'][index % batch_size] = mult_method
    df['mult_method_micro'][index % batch_size] = mult_method_micro
    df['mag_method_peak'][index % batch_size] = mag_method_peak
    df['mag_method'][index % batch_size] = mag_method
    df['mag_method_micro'][index % batch_size] = mag_method_micro
    df['coords'][index % batch_size] = coords
    df['obs_skybrightness'][index % batch_size] = obs_skybrightness
    df['obs_psf'][index % batch_size] = obs_psf
    df['obs_lim_mag'][index % batch_size] = obs_lim_mag
    df['obs_N_coadds'][index % batch_size] = obs_N_coadds
    df['survey'][index % batch_size] = survey
    df['rolling'][index % batch_size] = rolling
    df['obs_mag_micro'][index % batch_size] = obs_mag_micro
    df['mag_micro_error'][index % batch_size] = mag_micro_error
    df['obs_snr_micro'][index % batch_size] = obs_snr_micro
    df['mag_unresolved_micro'][index % batch_size] = mag_unresolved_micro
    df['mag_unresolved_micro_error'][index % batch_size] = mag_unresolved_micro_error
    df['snr_unresolved_micro'][index % batch_size] = snr_unresolved_micro
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


def _clean_obj_data(obj_data):
    """
    From Catarina Alves, to read in cadence dataframes.
    Make the event data consistent with snmachine.
    Parameters
    ----------
    obj_data :  astropy.table.table
        Observations of the event.
    Returns
    -------
    obj_data :  astropy.table.table
        Observations of the event.
    obj_name : str
        Name of the event.
    """

    # snmachine and SNANA use a different denomination
    obj_data.rename_columns(names=['FLUXCAL', 'FLUXCALERR', 'MJD'],
                            new_names=['flux', 'flux_error', 'mjd'])
    # Rename `filter` values as per `snmachine` convention
    obj_pb = list(obj_data['BAND'])
    obj_pb = [x.lower().strip() for x in obj_pb]
    obj_data['filter'] = obj_pb
    # Set detected flag in the observations; corresponds to SNANA flag 13
    is_detected = [('{0:020b}'.format(i))[-13]
                   for i in obj_data['PHOTFLAG']]
    is_detected = np.array(is_detected, dtype=int)
    obj_data['detected'] = np.array(is_detected, dtype=bool)
    # Add the object name to the light curve observations
    obj_name = obj_data.meta['SNID'].astype(str)
    obj_data['object_id'] = obj_name
    return obj_data


def main():

    print("test")


if __name__ == '__main__':
    main()

