#! /bin/python3
import numpy as np
import json
from functions import find_nearest
import sqlite3
import time
import matplotlib.pyplot as plt
import seaborn as sns
from lenstronomy.Util import param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from astropy.cosmology import FlatLambdaCDM

        
class Microlensing:

    def __init__(self, lens_model_class, kwargs_lens, x_image, y_image, theta_E, z_lens, z_source, cosmo, bandpasses):
        """
        This class computes the microlensing parameters kappa, gamma, and s, and calculates the microlensing
        contribution to the supernova light curve.

        :param lens_model_class: Lenstronomy object returned from LensModel containing the lens properties
        :param kwargs_lens: list of keyword arguments for the PEMD and external shear lens model
        :param x_image: numpy array of length [num_images] containing the x-positions of the SN images in arcsec
        :param y_image: numpy array of length [num_images] containing the y-positions of the SN images in arcsec
        :param theta_E: einstein radius of the lens galaxy (float)
        :param z_lens: redshift of the lens galaxy (float)
        :param z_source: redshift of the supernova (float)
        :param cosmo: instance of astropy containing the background cosmology
        :param bandpasses: list containing bandpasses that will be used, choose from 'g', 'r', 'i', 'z' and 'y'
        """

        self.lens_model_class = lens_model_class
        self.kwargs_lens = kwargs_lens
        self.x_image = x_image
        self.y_image = y_image
        self.theta_E = theta_E
        self.z_lens = z_lens
        self.z_source = z_source
        self.cosmo = cosmo
        self.bandpasses = bandpasses

    def get_kappa(self):
        """
        Calculate the convergence (surface mass density) kappa at the positions of the images.

        :return: array of length [num_images] containing the convergence/kappa for each image
        """
        kappa = self.lens_model_class.kappa(x=self.x_image, y=self.y_image, kwargs=self.kwargs_lens)
        return kappa

    def get_gamma(self):
        """
        Calculate the shear at the position of the images.

        :return: array of length [num_images] containing the shear/gamma for each image
        """
        gamma_scalar_A = ((self.lens_model_class.gamma(x=self.x_image[0], y=self.y_image[0],
                                                       kwargs=self.kwargs_lens)[0])**2 +
                          (self.lens_model_class.gamma(x=self.x_image[0], y=self.y_image[0],
                                                       kwargs=self.kwargs_lens)[1]) ** 2) ** 0.5
        gamma_scalar_B = ((self.lens_model_class.gamma(x=self.x_image[1], y=self.y_image[1],
                                                       kwargs=self.kwargs_lens)[0]) ** 2 +
                          (self.lens_model_class.gamma(x=self.x_image[1], y=self.y_image[1],
                                                       kwargs=self.kwargs_lens)[1]) ** 2) ** 0.5

        if len(self.x_image) == 4:
            gamma_scalar_C = ((self.lens_model_class.gamma(x=self.x_image[2], y=self.y_image[2],
                                                           kwargs=self.kwargs_lens)[0]) ** 2 +
                              (self.lens_model_class.gamma(x=self.x_image[2], y=self.y_image[2],
                                                           kwargs=self.kwargs_lens)[1]) ** 2) ** 0.5
            gamma_scalar_D = ((self.lens_model_class.gamma(x=self.x_image[3], y=self.y_image[3],
                                                           kwargs=self.kwargs_lens)[0]) ** 2 +
                              (self.lens_model_class.gamma(x=self.x_image[3], y=self.y_image[3],
                                                           kwargs=self.kwargs_lens)[1]) ** 2) ** 0.5
            return np.array([gamma_scalar_A, gamma_scalar_B, gamma_scalar_C, gamma_scalar_D])

        elif len(self.x_image) == 2:
            return np.array([gamma_scalar_A, gamma_scalar_B])

    def get_s(self, R_eff):
        """
        Calculates the smooth matter fraction s:  the fraction of the total matter that is in the form of dark matter.
        s = 1 - kappa_stellar/kappa_total.
        The stellar convergence is approximated by a Vaucouleurs profile, assuming spherical symmetry. The Vaucouleurs
        profile is normalised such that the stellar convergence is always lower than the total convergence.
        The effective radius of the lens galaxy is determined through the fundamental plane for elliptical galaxies
        (Hyde and Bernardi, 2009).
        The total convergence is obtained from Lenstronomy.

        :param R_eff: effective radius of the lens galaxy in arcsec (float)
        :return: array of length [num_images] containing the smooth matter fraction for each image
        """

        A = self.normalise_stellar_convergence(R_eff)
        if np.isnan(A):
            return np.nan

        radius = (self.x_image ** 2 + self.y_image ** 2) ** 0.5
        iso_coords = radius / np.sqrt(2)
        kappa_stellar = self.stellar_convergence(A, radius, R_eff)
        kappa_total = self.lens_model_class.kappa(x=iso_coords, y=iso_coords, kwargs=self.kwargs_lens)

        if len(self.x_image) == 2:
            return np.array([(1 - kappa_stellar[0] / kappa_total[0]), (1 - kappa_stellar[1] / kappa_total[1])])
        elif len(self.x_image) == 4:
            return np.array([(1 - kappa_stellar[0] / kappa_total[0]), (1 - kappa_stellar[1] / kappa_total[1]),
                             (1 - kappa_stellar[2] / kappa_total[2]), (1 - kappa_stellar[3] / kappa_total[3])])

    def stellar_convergence(self, A, radius, R_eff):
        """
        Calculates the stellar convergence from the position and effective radius with a Vaucouleurs profile,
        following Dobler & Keeton (2006): https://arxiv.org/pdf/astro-ph/0608391.pdf

        :param A: normalisation constant for the stellar convergence profile (float)
        :param radius: radius from the centre of the lens galaxy in arcsec (float)
        :param R_eff: effective radius of the lens galaxy in arcsec (float)
        :return: array of length [num_images] containing the stellar convergence at image positions
        """
        k = 7.67
        s_convergence = A * np.exp(-k * (radius / R_eff) ** (1 / 4))
        return s_convergence

    def radial_profiles(self, A, R_eff, arrays=False):
        """
        Compute the radial profiles for the stellar and total convergence.
        Return either radial profiles (arrays=True), or the maximum difference between the two (arrays=False).

        :param A: normalisation constant for the stellar convergence profile (float)
        :param R_eff: effective radius of the lens galaxy in arcsec (float)
        :param arrays: bool. if True, output radial profiles. if False, output maximum difference between the profiles
        :return: if arrays=True: R_profile: array of length 100 containing the range of radius values that are considered
                                 total_profile: array of length 100 containing the total radial convergence profile
                                 stellar_profile: array of length 100 containing the stellar radial convergence profile
                 if arrays=False: the maximum difference between the total and stellar convergence profiles (float)
        """
        R_profile = np.linspace(0.01, 2, 100)
        total_profile = []
        stellar_profile = []
        # SIS_profile = []

        for r in R_profile:
            x = r / np.sqrt(2)
            total_profile.append(self.lens_model_class.kappa(x=x, y=x, kwargs=self.kwargs_lens))
            stellar_profile.append(self.stellar_convergence(A, r, R_eff))
            # SIS_profile.append(self.theta_E / (2 * r))

        if arrays:
            # Return the radius-range, total convergence and stellar convergence profiles
            return R_profile, total_profile, stellar_profile

        # Return the maximum difference between stellar and total convergence profiles
        max_difference = max(np.array(stellar_profile) - np.array(total_profile))
        return max_difference

    def normalise_stellar_convergence(self, R_eff):
        """
        Normalization of stellar convergence profile constant A.
        Find A such that the stellar convergence is always smaller than the total convergence,
        The function self.radial_profiles computes (kappa_stellar - kappa_total), and this should be smaller than
        0 for all radii. In a tree-based manner, the starting value of A is determined at which to start scanning.

        :param R_eff: R_eff: effective radius of the lens galaxy in arcsec (float)
        :return: the value of the normalisation constant A (float), such that the stellar convergence is always smaller
                 than the total convergence
        """

        # Determine at which value of A to start sampling
        if self.radial_profiles(400, R_eff) > 0:
            if self.radial_profiles(200, R_eff) > 0:
                if self.radial_profiles(100, R_eff) > 0:
                    return np.nan
                else:
                    A_start = 100
            else:
                if self.radial_profiles(300, R_eff) > 0:
                    A_start = 200
                else:
                    A_start = 300
        else:
            if self.radial_profiles(600, R_eff) > 0:
                if self.radial_profiles(500, R_eff) > 0:
                    A_start = 400
                else:
                    A_start = 500
            else:
                if self.radial_profiles(800, R_eff) > 0:
                    if self.radial_profiles(700, R_eff) > 0:
                        A_start = 600
                    else:
                        A_start = 700
                else:
                    if self.radial_profiles(1000, R_eff) > 0:
                        if self.radial_profiles(900, R_eff) > 0:
                            A_start = 800
                        else:
                            A_start = 900
                    else:
                        A_start = 1000

        # Scan increasing values of A, until stellar_convergence > total_convergence
        A = A_start
        stepsize = 10
        while self.radial_profiles(A, R_eff) < 0:
            A += stepsize
        # Final value of A (go back one step):
        A -= stepsize

        return A

    def get_kgs(self, kappa, gamma, s):
        """
        For a lensed supernova image with given convergence (kappa), shear (gamma), and smooth matter fraction (s):
        select the combination of (kappa, gamma, s) from our simulation that is closest to those values.

        :param kappa: convergence for 1 image (float)
        :param gamma: shear for 1 image  (float)
        :param s: smooth matter fraction for 1 image  (float)
        :return: tuple containing (kappa, gamma s) to be used for the given lensed supernova image
        """

        kgs = [(3.620489580770965832e-01, 3.416429828804125046e-01, 4.430360463808165061e-01),
               (6.550590438288885764e-01, 6.694697862208409678e-01, 4.430360463808165061e-01),
               (6.550590438288885764e-01, 9.517424234642006819e-01, 4.430360463808165061e-01),
               (9.564962670984238358e-01, 6.694697862208409678e-01, 4.430360463808165061e-01),
               (9.564962670984238358e-01, 9.517424234642006819e-01, 4.430360463808165061e-01),
               (3.620489580770965832e-01, 3.416429828804125046e-01, 0.616),
               (6.550590438288885764e-01, 6.694697862208409678e-01, 0.616),
               (6.550590438288885764e-01, 9.517424234642006819e-01, 0.616),
               (9.564962670984238358e-01, 6.694697862208409678e-01, 0.616),
               (9.564962670984238358e-01, 9.517424234642006819e-01, 0.616),
               (3.620489580770965832e-01, 3.416429828804125046e-01, 7.902578031429109418e-01),
               (6.550590438288885764e-01, 6.694697862208409678e-01, 7.902578031429109418e-01),
               (6.550590438288885764e-01, 9.517424234642006819e-01, 7.902578031429109418e-01),
               (9.564962670984238358e-01, 6.694697862208409678e-01, 7.902578031429109418e-01),
               (9.564962670984238358e-01, 9.517424234642006819e-01, 7.902578031429109418e-01),
               (3.620489580770965832e-01, 2.800000000000000266e-01, 9.100000000000000311e-01)]

        distances = []
        for K, G, S in kgs:
            dist = ((K - kappa) ** 2 + (G - gamma) ** 2 + (S - s) ** 2) ** 0.5
            distances.append(dist)

        combination = kgs[np.argmin(distances)]
        return combination

    def get_kgs_all_images(self, kappa, gamma, s):
        """
        Returns the values of (kappa, gamma, s) for each image that correspond to the microlensing simulation.

        :param kappa: array of length [num_images] containing the convergence (kappa) for each image
        :param gamma: array of length [num_images] containing the shear (gamma) for each image
        :param s: array of length [num_images] containing the smooth matter fraction (s) for each image
        :return: list of length [num_images] containing numpy arrays of length 3 with the final values of kappa, gamma
                 and s for each image
        """

        final_kgs = []

        for i in range(len(self.x_image)):
            final_kgs.append(tuple(np.around(self.get_kgs(kappa[i], gamma[i], s[i]), 3)))

        return final_kgs

    def query_database(self, c, kappa, gamma, s, source_redshift, data_type, bandpass, SN_model):
        """
        Select one element (row) from the microlensing lightcurves database.

        :param c: cursor to the database
        :param kappa: convergence (float)
        :param gamma: shear (float)
        :param s: smooth matter fraction (float)
        :param source_redshift: redshift of the supernova, rounded to match the simulation redshifts (float)
        :param data_type: this refers to either a microlensing light curve ('micro'), a macrolensing light curve ('macro'),
               or the time sampling of the light curves ('time')
        :param bandpass: telescope filter used for simulation. choose from ["u","g","r","i","z","y","J","H"]
        :param SN_model: supernova explosion model used in the simulation. choose from ["m", "n", "w", "s"]
        :return: single entry (array) from the database, either a micro light curve, macro light curve, or time sampling
        """

        file_name = "kappa:%.3f_gamma:%.3f_s:%.3f_zsrc:%.2f" % (kappa, gamma, s, source_redshift)

        condition = "filename = :filename AND data_type = :data_type AND bandpass = :bandpass AND SN_model = :SN_model"
        condition_values = {'filename': file_name,
                            'data_type': data_type,
                            'bandpass': bandpass,
                            'SN_model': SN_model}

        #n1time = time.time()
        c.execute("SELECT * FROM datapoints WHERE " + condition + " ORDER BY RANDOM() LIMIT 1", condition_values)
        #n2time = time.time()
        #print(n2time - n1time)

        """
        else:

            database_name = '../data/microlensing/databases/microlensing_database_z_1_30.db'
            conn = sqlite3.connect(database_name)
            c = conn.cursor()
            file_name = "kappa:0.362_gamma:0.342_s:0.616_zsrc:1.30"

            n1time = time.time()
            line_id = np.random.randint(0, 5120528)
            print("line_id: ", line_id)
            condition = "filename = :filename AND line_id = :line_id"
            condition_values = {'filename': file_name,
                                'line_id': line_id}
            n2time = time.time()
            con_time = n2time - n1time
            #print("con_time: ", con_time)

            n1time = time.time()
            c.execute("SELECT * FROM datapoints WHERE " + condition, condition_values)
            n2time = time.time()
            #print((n2time - n1time) + con_time)
            result = c.fetchone()
            print(result)

            n1time = time.time()
            c.execute("SELECT * FROM datapoints WHERE " + condition + " LIMIT 1", condition_values)
            n2time = time.time()
            #print((n2time - n1time) + con_time)
            result = c.fetchone()
            print(result)

            # -----------------------
            n1time = time.time()
            condition = "filename = :filename AND data_type = :data_type AND SN_model = :SN_model"
            condition_values = {'filename': file_name,
                                'data_type': data_type,
                                'SN_model': SN_model}

            c.execute("SELECT * FROM datapoints WHERE " + condition + " ORDER BY RANDOM() LIMIT 1", condition_values)
            n2time = time.time()
            #print(n2time - n1time)
            # -----------------------

            print(" ")
        """

        result = c.fetchone()

        return result

    def micro_lightcurve(self, kgs, source_redshift, SN_model):
        """
        Fetches a random microlensing and macrolensing light curve from the database with (kappa, gamma, s) similar
        to the lensed supernova image.

        :param kgs: tuple containing the values of (kappa, gamma, s) for the given image
        :param source_redshift: redshift of the supernova, rounded to match the simulation redshifts
        :param SN_model: supernova explosion model used in the simulation. choose from ["m", "n", "w", "s"]
        :return: microlensing_contribution: array containing the microlensing contributions to the light curve for
                 the given image (in magnitudes)
                 macro_lightcurve: array containing the macrolensed light curve for a given image (in magnitudes)
                 micro_times: array containing the time stamps corresponding to the two arrays mentioned above
        """

        timing1_start = time.time()
        # =========================================== TIMING 1 ===========================================

        kappa, gamma, s = kgs
        database_name = '../data/microlensing/databases/microlensing_database_z_%i_%s.db' % (int(np.floor(source_redshift)),
                        np.char.zfill(str(int(np.around(100 * (source_redshift - int(np.floor(source_redshift)))))), 2))

        # =========================================== TIMING 1 ===========================================
        timing1_end = time.time()
        self.mmtiming1.append(timing1_end - timing1_start)

        timing2_start = time.time()
        # =========================================== TIMING 2 ===========================================

        # connect to the database and get a cursor
        conn = sqlite3.connect(database_name)
        c = conn.cursor()

        # =========================================== TIMING 2 ===========================================
        timing2_end = time.time()
        self.mmtiming2.append(timing2_end - timing2_start)

        timing3_start = time.time()
        # =========================================== TIMING 3 ===========================================


        micro = self.query_database(c, kappa, gamma, s, source_redshift, 'micro', 'i', SN_model)[5:]

        # =========================================== TIMING 3 ===========================================
        timing3_end = time.time()
        self.mmtiming3.append(timing3_end - timing3_start)

        timing4_start = time.time()
        # =========================================== TIMING 4 ===========================================


        macro = self.query_database(c, kappa, gamma, s, source_redshift, 'macro', 'i', SN_model)[5:]

        # =========================================== TIMING 4 ===========================================
        timing4_end = time.time()
        self.mmtiming4.append(timing4_end - timing4_start)

        timing5_start = time.time()
        # =========================================== TIMING 5 ===========================================

        time_range = self.query_database(c, kappa, gamma, s, source_redshift, 'time', 'none', 'none')[5:]

        # =========================================== TIMING 5 ===========================================
        timing5_end = time.time()
        self.mmtiming5.append(timing5_end - timing5_start)

        timing6_start = time.time()
        # =========================================== TIMING 6 ===========================================


        micro_contribution = np.array(micro) - np.array(macro)

        # =========================================== TIMING 6 ===========================================
        timing6_end = time.time()
        self.mmtiming6.append(timing6_end - timing6_start)

        return micro_contribution, np.array(macro), np.array(time_range)

    def micro_lightcurve_all_images(self, kappa, gamma, s):
        """
        Returns a list with microlensing contributions for each lensed supernova image.

        :param kappa: array of length [num_images] containing the convergence (kappa) for each image
        :param gamma: array of length [num_images] containing the shear (gamma) for each image
        :param s: array of length [num_images] containing the smooth matter fraction (s) for each image
        :return: micro_contributions: list of length [num_images] containing arrays with the microlensing contributions
                 to the light curve (in magnitudes)
                 macro_contributions: list of length [num_images] containing arrays with the macrolensed light curve
                 micro_times: list of length [num_images] containing arrays with the time stamps for the micro and
                 macrolensing light curves
        """

        mtiming1 = []
        mtiming2 = []

        self.mmtiming1 = []
        self.mmtiming2 = []
        self.mmtiming3 = []
        self.mmtiming4 = []
        self.mmtiming5 = []
        self.mmtiming6 = []
        self.mmtiming7 = []

        timing1_start = time.time()
        # =========================================== TIMING 1 ===========================================

        final_kgs = self.get_kgs_all_images(kappa, gamma, s)
        micro_z_source = np.around(find_nearest(np.arange(0, 1.45, 0.05), self.z_source), 2)[0]

        # Choose a random SN explosion model
        SN_model = np.random.choice(["m", "n", "w", "s"])
        if micro_z_source == 1.4:
            SN_model = "w"

        micro_contributions = []
        macro_contributions = []
        micro_times = []

        # =========================================== TIMING 1 ===========================================
        timing1_end = time.time()
        mtiming1.append(timing1_end - timing1_start)

        timing2_start = time.time()
        # =========================================== TIMING 2 ===========================================

        for i in range(len(self.x_image)):
            micro, macro, time_range = self.micro_lightcurve(final_kgs[i], micro_z_source, SN_model)
            micro_contributions.append(micro)
            macro_contributions.append(macro)
            micro_times.append(time_range)

        timing2_end = time.time()
        mtiming2.append(timing2_end - timing2_start)
        # =========================================== TIMING 2 ===========================================

        return micro_contributions, macro_contributions, micro_times, mtiming1, mtiming2, \
               self.mmtiming1, self.mmtiming2, self.mmtiming3, self.mmtiming4, self.mmtiming5, self.mmtiming6, self.mmtiming7,

    def micro_snapshot(self, micro_lightcurves, macro_lightcurves, micro_times, td_images, day, peak=False):
        """
        Determines the microlensing contribution to the light curve for one specific observation.

        :param micro_lightcurves: list of length [num_images] containing arrays with the microlensing contributions
                 to the light curve (in magnitudes)
        :param macro_lightcurves: list of length [num_images] containing arrays with the macrolensed light curve
        :param micro_times: list of length [num_images] containing arrays with the time stamps for the micro and
                 macrolensing light curves
        :param td_images: array of length [num_images] containing the time delays between the supernova images
        :param day: time stamp corresponding to the observation (in days, float)
        :param peak: bool. if True, calculate microlensing contribution at the peak for each image. if False,
                 take into account time delays and compute the contribution at time 'day'
        :return: array of length [num_images] containing the microlensing contribution corresponding to the
                 observational time stamp as given by 'day'
        """

        # plt.figure()
        micro_day = []

        for i in range(len(self.x_image)):
            micro_peak = micro_times[i][np.argmin(macro_lightcurves[i])]
            micro_times[i] -= micro_peak
            if not peak:
                micro_times[i] += td_images[i]

            t0, index_t0 = find_nearest(micro_times[i], day)
            micro_t0 = micro_lightcurves[i][index_t0]
            micro_day.append(micro_t0)

            # plt.plot(micro_times[i], macro_lightcurves[i], '.')
            # plt.axvline(x=day, ls='--', color='gray', lw=1)
            # plt.plot(t0, macro_lightcurves[i][index_t0] + micro_t0, 'o', color='red', ms=10)

        return micro_day



def main():

    print("Test")


if __name__ == '__main__':
    main()


    """
    def micro_lightcurve_old(self, kgs, source_redshift):

        # Directory containing the light curves
        output_data_path = "microlensing/light_curves/"

        # Light curve properties
        kappa, gamma, s = kgs
        lens_redshift = 0.32
        N_sim = 10000

        # Open corresponding pickle file with light curve
        pickel_name = "k%f_g%f_s%.3f_redshift_source_%.3f_lens%.3f_Nsim_%i" % (
        kappa, gamma, s, source_redshift, lens_redshift, N_sim)

        open_pickle = "%s%s.pickle" % (output_data_path, pickel_name)
        with open(open_pickle, 'r') as handle:
            d_light_curves = json.load(handle, encoding='latin1')

        # Get Macrolensed light curve and time steps
        # Filter options: ["u","g","r","i","z","y","J","H"]
        SN_model = "ww"  # Options: ["me", "n1", "ww", "su"]
        key_macro = "macro_light_curve_%s%s" % (SN_model, self.bandpass)
        macro_light_curve = d_light_curves[key_macro]
        time_after_explosion = d_light_curves["time_bin_center"]

        # get microlensed light curves
        micro_config = np.random.randint(0, 10000)
        key_micro = "micro_light_curve_%s%i%s" % (SN_model, micro_config, self.bandpass)
        micro_light_curve = d_light_curves[key_micro]

        # print("Len: ", len(d_light_curves.keys()))

        microlensing_contribution = np.array(micro_light_curve) - np.array(macro_light_curve)

        return microlensing_contribution, np.array(macro_light_curve), np.array(time_after_explosion)
    """
