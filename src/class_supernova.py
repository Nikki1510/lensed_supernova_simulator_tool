#! /bin/python3
import numpy as np
from lenstronomy.Util import param_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from astropy.cosmology import FlatLambdaCDM
import scipy.stats as stats
import sncosmo


class Supernova:

    def __init__(self, theta_E, z_lens, z_source, cosmo, source_x, source_y):
        """
        This class defines the supernova whose light is gravitationally lensed by the lens galaxy.

        :param theta_E: einstein radius of the lens galaxy (float)
        :param z_lens: redshift of the lens galaxy (float)
        :param z_source: redshift of the supernova (float)
        :param cosmo: instance of astropy containing the background cosmology
        :param source_x: x-position of the supernova relative to the lens galaxy in arcsec (float)
        :param source_y: y-position of the supernova relative to the lens galaxy in arcsec (float)
        """

        self.theta_E = theta_E
        self.z_lens = z_lens
        self.z_source = z_source
        self.cosmo = cosmo
        self.source_x = source_x
        self.source_y = source_y

    def get_image_pos_magnification(self, lens_model_class, kwargs_lens, min_distance, search_window):
        """
        Calculates the image positions and magnifications using Lenstronomy functions.

        :param lens_model_class: Lenstronomy object returned from LensModel
        :param kwargs_lens: list of keyword arguments for the PEMD and external shear lens model
        :param min_distance: smallest distance ... equal to pixel size of telescope
        :param search_window: area in array to search for SN images
        :return: two numpy arrays of length [num_images] containing the image positions in arcsec,
                 one numpy array of length [num_images] with the macro magnification for each image
        """
        lensEquationSolver = LensEquationSolver(lens_model_class)
        x_image, y_image = lensEquationSolver.findBrightImage(self.source_x, self.source_y, kwargs_lens,
                                                              min_distance=min_distance, search_window=search_window)

        macro_mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
        macro_mag = np.abs(macro_mag)
        return x_image, y_image, macro_mag

    def host_light_model(self):
        """
        Uses lenstronomy to calculate the light model of the supernova host galaxy.
        Note: in our current set-up (using difference imageging), there is no light contribution from the host galaxy.

        :return: source_model_class: Lenstronomy object returned from LightModel corresponding to the host galaxy
                 kwargs_source: list of keywords arguments for the host light model
        """
        # No host light model in difference images
        source_model_list = ['SERSIC_ELLIPSE']
        phi_source, q_source = np.random.uniform(-np.pi / 2, np.pi / 2), np.random.uniform(0.2, 0.9)
        # if q_source not in [0,1]: continue
        e1, e2 = param_util.phi_q2_ellipticity(phi_source, q_source)
        kwargs_sersic_source = {'amp': 0, 'R_sersic': 0.1, 'n_sersic': 2, 'e1': e1, 'e2': e2,
                                'center_x': self.source_x, 'center_y': self.source_y}
        kwargs_source = [kwargs_sersic_source]
        source_model_class = LightModel(light_model_list=source_model_list)
        return source_model_class, kwargs_source

    def separation(self, x_image, y_image):
        """
        Calculates the maximum image separation.

        :param x_image: array of length [num_images] containing the x coordinates of the supernova images in arcsec
        :param y_image: array of length [num_images] containing the y coordinates of the supernova images in arcsec
        :return: maximum separation between images in arcsec (float)
        """
        if len(x_image) == 2:
            sep = ((x_image[0] - x_image[1])**2 + (y_image[0] - y_image[1])**2)**0.5
            return sep
        elif len(x_image) == 4:
            sep1 = ((x_image[0] - x_image[1]) ** 2 + (y_image[0] - y_image[1]) ** 2) ** 0.5
            sep2 = ((x_image[0] - x_image[2]) ** 2 + (y_image[0] - y_image[2]) ** 2) ** 0.5
            sep3 = ((x_image[0] - x_image[3]) ** 2 + (y_image[0] - y_image[3]) ** 2) ** 0.5
            sep4 = ((x_image[1] - x_image[2]) ** 2 + (y_image[1] - y_image[2]) ** 2) ** 0.5
            sep5 = ((x_image[1] - x_image[3]) ** 2 + (y_image[1] - y_image[3]) ** 2) ** 0.5
            sep6 = ((x_image[2] - x_image[3]) ** 2 + (y_image[2] - y_image[3]) ** 2) ** 0.5
            separations = np.array([sep1, sep2, sep3, sep4, sep5, sep6])
            max_sep = max(separations)
            return max_sep

    def get_sncosmo_filter(self, telescope, band):
        """
        Determines the name of the SNcosmo filter corresponding to a specific telescope and bandpass.

        :param telescope: Choose 'LSST' or 'ZTF'
        :param band: bandpass, choose between 'g', 'r', 'i', 'z', 'y' for LSST and 'g', 'r', 'i' for ZTF.
        :return: string containing SNcosmo name for the desired filter/bandpass.
        """

        if telescope == 'LSST':
            if band == 'g':
                sncosmo_filter = 'lsstg'
            elif band == 'r':
                sncosmo_filter = 'lsstr'
            elif band == 'i':
                sncosmo_filter = 'lssti'
            elif band == 'z':
                sncosmo_filter = 'lsstz'
            elif band == 'y':
                sncosmo_filter = 'lssty'

        # Create elif telescope == 'ZTF'

        return sncosmo_filter

    def flux_ratio(self, model, macro_mag, micro_peak, telescope, band, add_microlensing):
        """
        Calculate the flux ratio between the brightest and faintest image.

        :param model: SNcosmo model for the supernova light curve
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param micro_peak: array of length [num_images] containing the microlensing contributions at peak
        :param telescope: choose 'LSST' or 'ZTF'
        :param band: bandpass, choose between 'g', 'r', 'i', 'z', 'y' for LSST and 'g', 'r', 'i' for ZTF.
        :param add_microlensing: bool. if False: only compute macro magnification. if True: also add microlensing
                contributions to the light curves
        :return: flux ratio between the brightest and faintest image (float)
        """

        sncosmo_filter = self.get_sncosmo_filter(telescope.telescope, band)

        if add_microlensing:
            zeropoint = sncosmo.ABMagSystem(name='ab').zpbandflux(sncosmo_filter)
            magnitude_macro = model.bandmag(sncosmo_filter, time=0, magsys='ab')
            magnitude_macro -= 2.5 * np.log10(macro_mag)
            magnitude_micro = magnitude_macro + micro_peak
            flux = 10**(magnitude_micro / -2.5) * zeropoint

        else:
            flux = model.bandflux(sncosmo_filter, time=0)
            flux *= macro_mag

        flux_ratio = flux[1] / flux[0]
        return flux_ratio

    def check_detectability_peak(self, telescope, model, macro_mag, micro_peak, add_microlensing):
        """
        Check whether the lensed SN peak brightness can be detected by the telescope at peak brightness.
        If it is detectable in any filter, return True.

        :param telescope: choose between 'LSST' and 'ZTF'
        :param model: SNcosmo model for the supernova light curve
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param micro_peak: array of length [num_images] containing the microlensing contributions at peak
        :param add_microlensing: bool. if False: no microlensing. if True: also add microlensing to the peak
        :return: bool. True: detectable in at least 1 filter. False: not detectable in any filter.
        """

        for band in telescope.bandpasses:

            num_images = len(macro_mag)

            if num_images == 2:
                # Check if the flux ratio is between 0.1 and 10
                flux_ratio = self.flux_ratio(model, macro_mag, micro_peak, telescope, band, add_microlensing)
                if flux_ratio < 0.1 or flux_ratio > 10:
                    continue

            sncosmo_filter = self.get_sncosmo_filter(telescope.telescope, band)

            peak_brightness = model.bandmag(sncosmo_filter, time=0, magsys='ab')
            peak_brightness -= 2.5 * np.log10(macro_mag)

            if add_microlensing:
                peak_brightness += micro_peak

            limiting_magnitude = telescope.single_band_properties(band)[1]

            if num_images == 2:
                if len(peak_brightness[peak_brightness < limiting_magnitude]) == 2:
                    return True
            elif num_images == 4:
                if len(peak_brightness[peak_brightness < limiting_magnitude]) >= 3:
                    return True
        return False

    def check_detectability(self, telescope, model, macro_mag, brightness_obs, obs_filters, micro_peak, add_microlensing):
        """
        Check whether the lensed SN peak brightness and flux ratio can be detected by the telescope.
        If it is detectable in any filter, return True.

        :param telescope: choose between 'LSST' and 'ZTF'
        :param model: SNcosmo model for the supernova light curve
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param brightness_obs: array of length [num_observations, num_images] containing the brightness of each observation
        :param obs_filters: array of length [num_observations] containing the filter used for each observation
        :param micro_peak: array of length [num_images] containing the microlensing contributions at peak
        :param add_microlensing: bool. if False: no microlensing. if True: also add microlensing to the peak
        :return: bool. True: detectable in at least 1 filter. False: not detectable in any filter.
        """

        for band in telescope.bandpasses:

            num_images = len(macro_mag)

            if num_images == 2:
                # Check if the flux ratio is between 0.1 and 10
                # Note: this only checks the flux ratio at peak, not for every image!
                flux_ratio = self.flux_ratio(model, macro_mag, micro_peak, telescope, band, add_microlensing)
                if flux_ratio < 0.1 or flux_ratio > 10:
                    continue

            # Check if the brightest image is brighter than the limiting magnitude
            indices = np.where(np.array(obs_filters) == band)[0]
            if len(indices) == 0:
                continue
            max_brightness = np.min(brightness_obs[indices], axis=0)

            limiting_magnitude = telescope.single_band_properties(band)[1]

            if num_images == 2:
                if len(max_brightness[max_brightness < limiting_magnitude]) == 2:
                    return True
            elif num_images == 4:
                if len(max_brightness[max_brightness < limiting_magnitude]) >= 3:
                    return True
        return False

    def brightest_obs_bands(self, telescope, macro_mag, brightness_obs, obs_filters):
        """
        Compute for each band the magnitudes of the brightest observation.

        :param telescope: choose between 'LSST' and 'ZTF'
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param brightness_obs: array of length [num_observations, num_images] containing the brightness of each observation
        :param obs_filters: array of length [num_observations] containing the filter used for each observation
        :return: array of [num_filters, num_images] containing the peak brightness for each filter/bandpass.
                 if no observations are made in a filter, the array contains np.nans
        """

        num_images = len(macro_mag)
        obs_peak = np.zeros((len(telescope.bandpasses), num_images))

        for b in range(len(telescope.bandpasses)):

            band = telescope.bandpasses[b]

            indices = np.where(np.array(obs_filters) == band)[0]

            # If no observations in this band
            if len(indices) == 0:
                obs_peak[b] = np.nan * np.ones(num_images)
                continue

            # Save the brightest observations in the band
            obs_peak[b] = np.min(brightness_obs[indices], axis=0)

        return obs_peak

    def M_obs(self, x1, c, M_corrected):
        """
        Applies the Tripp Formula to correlate absolute magnitude with colour and stretch.
        Values for Alpha and Beta are taken from Scolnic & Kessler (2016).

        :param x1: stretch parameter for the SALT3 model
        :param c: colour parameter for the SALT3 model
        :param M_corrected: The absolute magnitude of the supernova without stretch and colour correlations
        :return: M_observed: The absolute magnitude of the supernova correlated with its stretch and colour
        """
        Alpha = 0.14
        Beta = 3.1
        M_observed = - Alpha * x1 + Beta * c + M_corrected
        return M_observed

    def light_curve(self, z_source):
        """
        Samples light curve parameters from stretch and colour distributions based on Scolnic & Kessler (2016)
        and returns a type Ia SN light curve model from SNcosmo using the SALT3 model.

        :param z_source: redshift of the supernova
        :return: SNcosmo model of the supernova light curve, stretch parameter, colour parameter, MW dust contribution,
                 observed absolute magnitude
        """
        H_0 = self.cosmo.H(0).value
        dustmodel = sncosmo.F99Dust(r_v=3.1)
        model = sncosmo.Model(source='salt3', effects=[dustmodel], effect_names=['mw'], effect_frames=['obs'])
        H_0_fid = 74.03                                      # SH0ES 2019 value
        M_fid = -19.24                                       # Absolute magnitude corresponding to SH0ES 2019 H0 value

        MW_dust = np.random.uniform(0, 0.2)                  # Sample E(B-V)
        x1 = stats.skewnorm.rvs(-8.241, 1.2311, 1.6712)      # Sample stretch parameter
        c = stats.skewnorm.rvs(2.483, -0.08938, 0.1215)      # Sample colour parameter
        M_cosmo = 5 * np.log10(H_0 / H_0_fid) + M_fid        # Cosmology correction to absolute magnitude
        M_corrected = np.random.normal(M_cosmo, 0.12)        # Absolute magnitude without colour/stretch correlation
        M_observed = self.M_obs(x1, c, M_corrected)          # Absolute magnitude with colour/stretch correlation

        model.set(z=z_source, t0=0.0, x1=x1, c=c, mwebv=MW_dust)
        model.set_source_peakabsmag(M_observed, 'bessellb', 'ab', cosmo=self.cosmo)
        x0 = model.get('x0')
        model.set(x0=x0)
        return model, x1, c, MW_dust, M_observed

    def get_app_magnitude(self, model, day, macro_mag, td_images, micro_day, telescope, band, add_microlensing):
        """
        Calculate the apparent magnitude for each supernova image at a certain time stamp.

        :param model: SNcosmo model for the supernova light curve
        :param day: time stamp of observation
        :param macro_mag: array of length [num_images] containing the macro magnification of each image
        :param td_images: array of length [num_images] containing the time delays between the supernova images
        :param micro_day: array of length [num_images] containing the microlensing contribution corresponding to the
        :param telescope: choose 'LSST' or 'ZTF'
        :param band: bandpass, choose between 'g', 'r', 'i', 'z', 'y' for LSST and 'g', 'r', 'i' for ZTF.
        :param add_microlensing: bool. if False: only compute macro magnification. if True: also add microlensing
                contributions to the light curves
        :return: app_mag_ps: array of length [num_images] containing the apparent magnitude for each image
                 # new_peak_brightness_image: array of length [num_images] containing the currently brightest observations
        """

        sncosmo_filter = self.get_sncosmo_filter(telescope, band)

        app_mag_ps = model.bandmag(sncosmo_filter, time=day - td_images, magsys='ab')
        app_mag_ps = np.nan_to_num(app_mag_ps, nan=np.inf)
        app_mag_ps -= 2.5 * np.log10(macro_mag)
        if add_microlensing:
            app_mag_ps += micro_day

        # new_peak_brightness_image = np.minimum(peak_brightness_image, app_mag_ps)
        return app_mag_ps  # , new_peak_brightness_image



def main():

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    sn = Supernova(1.0, 0.3, 0.8, cosmo, 0.1, 0.2)
    print("SN redshift: ", sn.z_source)
    print(sn.light_curve(1))


if __name__ == '__main__':
    main()
