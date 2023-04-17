#! /bin/python3
import numpy as np
from lenstronomy.Util import param_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from astropy.cosmology import FlatLambdaCDM
import scipy.stats as stats
import sncosmo


class Supernova_Unlensed:

    def __init__(self, cosmo):
        """
        This class defines the supernova whose light is gravitationally lensed by the lens galaxy.

        :param cosmo: instance of astropy containing the background cosmology
        """

        self.cosmo = cosmo

        # Sample supernova redshift from distribution for unlensed supernovae
        z_range, z_rate = self.unlensed_redshift_distribution()
        self.z_source = np.random.choice(z_range, p=z_rate)

    def unlensed_redshift_distribution(self, zmax=1.2):
        """
        Create a distribution for the unlensed supernova rates.
        :param zmax: maximum value of redshifts sampled
        :return: redshift values and corresponding probabilities
        """

        z = np.linspace(0, zmax, 100)
        rate = 2.5 * 10 ** -5 * (1 + z) ** 1.5
        volume = self.cosmo.comoving_volume(z).to("Mpc**3").value
        z_rate = volume * rate
        rates = np.diff(z_rate)
        pdf = rates / np.nansum(rates)
        x = (z[:-1] + z[1:]) * .5
        return x, pdf

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
                                'center_x': 0.0, 'center_y': 0.0}
        kwargs_source = [kwargs_sersic_source]
        source_model_class = LightModel(light_model_list=source_model_list)
        return source_model_class, kwargs_source

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

    def brightest_obs_bands(self, telescope, obs_mag, obs_filters):
        """
        Compute for each band the magnitudes of the brightest observation.

        :param telescope: choose between 'LSST' and 'ZTF'
        :param brightness_obs: array of length [num_observations, num_images] containing the brightness of each observation
        :param obs_filters: array of length [num_observations] containing the filter used for each observation
        :return: array of [num_filters, num_images] containing the peak brightness for each filter/bandpass.
                 if no observations are made in a filter, the array contains np.nans
        """

        obs_peak = np.zeros(len(telescope.bandpasses))

        for b in range(len(telescope.bandpasses)):

            band = telescope.bandpasses[b]

            indices = np.where(np.array(obs_filters) == band)[0]

            # If no observations in this band
            if len(indices) == 0:
                obs_peak[b] = np.nan
                continue

            # Save the brightest observations in the band
            obs_peak[b] = np.min(obs_mag[indices])

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

    def get_app_magnitude(self, model, day, telescope_class, band, lim_mag):
        """
        Calculate the apparent magnitude + error for each supernova image at a certain time stamp.

        :param model: SNcosmo model for the supernova light curve
        :param day: time stamp of observation
        :param band: bandpass, choose between 'g', 'r', 'i', 'z', 'y' for LSST and 'g', 'r', 'i' for ZTF.
        :param lim_mag: limiting magnitude of the specific observation in the specific band (takes into account weather)
        :return: app_mag_model: array of length [num_images] containing the apparent magnitude from the model
                 app_mag_obs: array of length [num_images] containing the observed (perturbed) apparent magnitude
                 app_mag_error: array of length [num_images] containing the apparent magnitude error
        """

        sncosmo_filter = self.get_sncosmo_filter(telescope_class.telescope, band)

        zeropoint = telescope_class.single_band_properties(band)[3]

        lim_flux = 10**((zeropoint - lim_mag)/2.5)
        flux_error = lim_flux / 5

        flux_ps = model.bandflux(sncosmo_filter, time=day, zp=zeropoint, zpsys='ab')

        # Perturb the flux according to the flux error
        new_flux_ps = np.random.normal(loc=flux_ps, scale=abs(flux_error))
        if new_flux_ps < 0.0 or flux_ps < flux_error:
            new_flux_ps = 0.0

        # Calculate S/N
        snr = new_flux_ps / flux_error

        # Convert to magnitudes
        app_mag_model = zeropoint - 2.5 * np.log10(flux_ps)
        app_mag_obs = zeropoint - 2.5*np.log10(new_flux_ps)
        app_mag_obs = np.nan_to_num(app_mag_obs, nan=np.inf)
        app_mag_error = abs(-2.5 * flux_error / (new_flux_ps * np.log(10)))

        if app_mag_obs > 50:
            app_mag_obs = np.inf
            app_mag_error = np.nan

        return app_mag_model, app_mag_obs, app_mag_error, snr


def main():

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    sn = Supernova(1.0, 0.3, 0.8, cosmo, 0.1, 0.2)
    print("SN redshift: ", sn.z_source)
    print(sn.light_curve(1))


if __name__ == '__main__':
    main()
