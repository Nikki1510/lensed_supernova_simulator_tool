#! /bin/python3
import numpy as np
from lenstronomy.Util import param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
from scipy.interpolate import interp1d

        
class Lens:

    def __init__(self, theta_E, z_lens, z_source, cosmo):
        """
        This class defines the lens galaxy that is responsible for the gravitational lensing.

        :param theta_E: einstein radius of the lens galaxy (float)
        :param z_lens: redshift of the lens galaxy (float)
        :param z_source: redshift of the supernova behind the lens galaxy (float)
        :param cosmo: instance of astropy containing the background cosmology
        """

        self.theta_E = theta_E
        self.z_lens = z_lens
        self.z_source = z_source
        self.cosmo = cosmo

    def mass_model(self, model='PEMD'):
        """
        Uses Lenstronomy to calculate the mass model of the lens galaxy.

        :return: lens_model_class: Lenstronomy object returned from LensModel
                 kwargs_lens: list of keyword arguments for the PEMD and external shear lens model
                 gamma_lens: power-law slope of the lens mass profile (float)
                 e1_lens: ellipticity component of the lens mass profile (float)
                 e2_lens: ellipticity component of the lens mass profile (float)
                 gamma1: component of external shear (float)
                 gamma2: component of external shear (float)
        """
        # Lens model
        if model == 'PEMD':
            lens_model_list = ['PEMD', 'SHEAR']
        elif model == 'SIE':
            lens_model_list = ['SIE', 'SHEAR']
        phi_lens, q_lens = np.random.uniform(-np.pi / 2, np.pi / 2), np.random.normal(0.7, 0.15)
        if q_lens < 0 or q_lens > 1:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        e1_lens, e2_lens = param_util.phi_q2_ellipticity(phi_lens, q_lens)

        if model == 'PEMD':
            gamma_lens = np.random.normal(2.0, 0.1)
            kwargs_spemd = {'theta_E': self.theta_E, 'gamma': gamma_lens, 'center_x': 0.0, 'center_y': 0.0,
                            'e1': e1_lens, 'e2': e2_lens}
        elif model == 'SIE':
            gamma_lens = 2.0
            kwargs_spemd = {'theta_E': self.theta_E, 'center_x': 0.0, 'center_y': 0.0,
                            'e1': e1_lens, 'e2': e2_lens}

        # External shear
        gamma1, gamma2 = param_util.shear_polar2cartesian(phi=np.random.uniform(-np.pi / 2, np.pi / 2),
                                                          gamma=np.random.uniform(0, 0.05))
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2}

        kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list, z_lens=self.z_lens,
                                     z_source=self.z_source, cosmo=self.cosmo)
        return lens_model_class, kwargs_lens, gamma_lens, e1_lens, e2_lens, gamma1, gamma2


    def light_model(self):
        """
        Uses lenstronomy to calculate the light model of the lens galaxy.
        Note: in our current set-up (using difference imageging), there is no light contribution from the lens galaxy.

        :return: lens_light_model_class: Lenstronomy object returned from LightModel corresponding to the lens galaxy
                 kwargs_lens_light: list of keywords arguments for the lens light model
        """
        # No lens light model in difference images
        lens_light_model_list = ['SERSIC_ELLIPSE']
        phi_light, q_light = np.random.uniform(-np.pi / 2, np.pi / 2), np.random.normal(0.7, 0.15)
        # if q_light not in [0,1]: continue
        e1, e2 = param_util.phi_q2_ellipticity(phi_light, q_light)
        kwargs_sersic_lens = {'amp': 0, 'R_sersic': 1, 'n_sersic': 2,
                              'e1': e1, 'e2': e2, 'center_x': 0.0, 'center_y': 0}
        kwargs_lens_light = [kwargs_sersic_lens]
        lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
        return lens_light_model_class, kwargs_lens_light

    def time_delays(self, lens_model_class, kwargs_lens, x_image, y_image):
        """
        Returns the differences in arrival times for each supernova image, calculated by Lenstronomy.

        :param lens_model_class: Lenstronomy object returned from LensModel
        :param kwargs_lens: list of keyword arguments for the PEMD and external shear lens model
        :param x_image: array of length [num_images] containing the x coordinates of the supernova images in arcsec
        :param y_image: array of length [num_images] containing the y coordinates of the supernova images in arcsec
        :return: array of length [num_images] containing the relative time delays between the images
        """
        td_images = lens_model_class.arrival_time(x_image=x_image, y_image=y_image, kwargs_lens=kwargs_lens)
        td_images = td_images - min(td_images)
        return td_images

    def velocity_dispersion(self):
        """
        Calculates the velocity dispersion from the Einstein radius and angular diameter distances.

        :return: velocity dispersion in km/s (float)
        """
        c = 299792.458  # km/s
        D_ls = self.cosmo.angular_diameter_distance_z1z2(self.z_lens, self.z_source).value
        D_s = self.cosmo.angular_diameter_distance(self.z_source).value
        theta_E_radians = self.theta_E * 2 * np.pi / 360 / 3600
        sigma = (c ** 2 * theta_E_radians * D_s / (4 * np.pi * D_ls)) ** 0.5
        return sigma

    def k_correction(self, redshift):
        """
        Uses the approximate k-correction for elliptical galaxies from Collett (2015)
        https://github.com/tcollett/LensPop
        Interpolates between the (redshift, k-correction) points using scipy.interp1d.

        :param redshift: the input lens redshift for which a k-correction needs to be calculated
        :return: the k-correction for the input redshift
        """

        colours = pd.read_pickle('../data/lenspopsplines.pkl')
        bands = colours[-2]
        z = bands['i_SDSS'][0][bands['i_SDSS'][1] != 0.0]
        kcor = bands['i_SDSS'][1][bands['i_SDSS'][1] != 0.0]
        kcorrection = interp1d(z, kcor, bounds_error=False, fill_value='extrapolate')
        return kcorrection(redshift)

    def scaling_relations(self):
        """
        Calculates the effective radius and r-band absolute magnitude from the velocity dispersion,
        following Hyde & Bernardi (2009) - "Curvature in the scaling relations of early-type galaxies".
        See also LensPop from Collett (2015): https://github.com/tcollett/LensPop

        :return: m_i: apparent magnitude of the lens galaxy in the i-band (float),
                 r_arcsec: effective radius of the lens galaxy in arcsec (float)
        """

        # Scaling relation for absolute r-band magnitude
        sigma = self.velocity_dispersion()
        V = np.log10(sigma)
        Mr = (-0.37 + (0.37 ** 2 - (4 * 0.006 * (2.97 + V))) ** 0.5) / (2 * 0.006)
        Mr += np.random.randn() * (0.15 / 2.4)

        # Apply k-correction to obtain m_i from M_r
        k_corr = self.k_correction(self.z_lens)
        m_i = Mr + self.cosmo.distmod(self.z_lens).value - k_corr

        # Scaling relation for effective radius
        R = 2.46 - 2.79 * V + 0.84 * V ** 2
        R += np.random.randn() * 0.11
        H_0 = self.cosmo.H(0).value
        r_phys = 10 ** R * H_0 / 100
        r_rad = r_phys / (self.cosmo.angular_diameter_distance(self.z_lens).value * 10 ** 3)
        r_arcsec = r_rad * 360 / (2 * np.pi) * 3600

        return m_i, r_arcsec


def main():

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    lens = Lens(1.0, 0.3, 0.8, cosmo)
    print("Lens redshift: ", lens.z_lens)


if __name__ == '__main__':
    main()

