#! /bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import sncosmo
import lenstronomy.Util.util as util
from lenstronomy.Util import constants as const
plt.rc("font", family="serif")
plt.rc("text", usetex=True)

        
class Visualisation:

    def __init__(self, time_delay_distance, td_images, theta_E, data_class, macro_mag, days):
        """
        This class visualises a lens system by printing out its properties and by plotting the time delay surface,
        light curves and observations.

        :param time_delay_distance: time delay distance of the lens system in Mpc (float)
        :param td_images: array of length [num_images] containing the relative time delays between the supernova images
        :param theta_E: einstein radius of the lens galaxy (float)
        :param data_class: instance of ImageData() from Lenstronomy containing the image properties
        :param macro_mag: array of length [num_images] with the macro magnification for each image
        :param days: array containing the time stamps (in days) of the observations
        """

        self.time_delay_distance = time_delay_distance
        self.td_images = td_images
        self.theta_E = theta_E
        self.macro_mag = macro_mag
        self.days = days
        self.data_class = data_class

        self.max_coordinate, self.min_coordinate = max(self.data_class.pixel_coordinates[0][0]), \
                                                   min(self.data_class.pixel_coordinates[0][0])

    def print_properties(self, peak_brightness_image, z_lens, z_source, H_0, micro_peak):
        """
        Prints out the key properties of the lensed supernova system.

        :param peak_brightness_image: array of length [num_images] containing the brightest apparent magnitudes of each
               of the supernova images
        :param z_lens: redshift of the lens galaxy (float)
        :param z_source: redshift of the supernova (float)
        :param H_0: value of the Hubble constant used for the background cosmology in this current lens configuration (float)
        :param micro_peak: array of length [num_images] containing the microlensing contributions at light curve peak
        :return: printed statements about the key properties of the lens system
        """
        print(" ")
        print("Peak brightness images:", peak_brightness_image)
        print("Time delays: ", self.td_images, "days")
        print("Macro magnification: ", self.macro_mag)
        print("Microlensing contribution at peak (magnitudes): ", np.around(micro_peak, 2))
        print("Lens redshift: ", z_lens)
        print("Supernova redshift: ", z_source)
        print("Einstein radius: ", self.theta_E)
        print("Hubble constant: ", H_0)
        print("Time-delay distance: ", self.time_delay_distance)
        print(" ")

    def plot_td_surface(self, lens_model_class, kwargs_lens, source_x, source_y, x_image, y_image):
        """
        Plots the time delay surface, which visualises the difference in arrival time for each point in space.
        The images (indicated as black dots) are formed on the extrema of the time delay surface.

        :param lens_model_class: Lenstronomy object returned from LensModel
        :param kwargs_lens: list of keyword arguments for the PEMD and external shear lens model
        :param source_x: x-position of the supernova relative to the lens galaxy in arcsec (float)
        :param source_y: y-position of the supernova relative to the lens galaxy in arcsec (float)
        :param x_image: array of length [num_images] containing the x coordinates of the supernova images in arcsec
        :param y_image: array of length [num_images] containing the y coordinates of the supernova images in arcsec
        :return: plots the fermat surface/ time delay surface in days
        """
        x_grid, y_grid = self.data_class.pixel_coordinates
        x_grid1d = util.image2array(x_grid)
        y_grid1d = util.image2array(y_grid)

        # Calculate fermat surface
        fermat_surface = lens_model_class.fermat_potential(x_image=x_grid1d, y_image=y_grid1d,
                                                           kwargs_lens=kwargs_lens, x_source=source_x,
                                                           y_source=source_y)
        fermat_surface = util.array2image(fermat_surface)

        # Convert to units of days
        fermat_surface = fermat_surface * self.time_delay_distance * const.Mpc / const.day_s * const.arcsec**2 / const.c

        # Create figure
        fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))

        # Colour background according to fermat surface
        im1 = ax1.matshow(np.log(fermat_surface + 10 + abs(np.min(fermat_surface))), origin='lower', cmap='viridis_r',
                          extent=[self.min_coordinate, self.max_coordinate, self.min_coordinate, self.max_coordinate])
        cbar1 = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label(r'days', fontsize=18)

        # Draw contours following the fermat surface
        ax1.contour(x_grid, y_grid, fermat_surface, colors='white', linestyles='-', alpha=0.5, origin='lower',
                    levels=np.linspace(start=np.min(fermat_surface), stop=max(self.td_images), num=5))

        # Plot the position of the images
        ax1.scatter(x_image, y_image, s=150, edgecolors="white", facecolors='black')

        ax1.set_ylabel("arcsec", fontsize=18)
        ax1.set_xlabel("arcsec", fontsize=18)
        ax1.xaxis.set_ticks_position('bottom')
        ax1.set_title("Time delay surface", fontsize=25)

        # Scale figure according to Einstein radius
        if 5 * self.theta_E < self.max_coordinate:
            ax1.set_xlim(-5 * self.theta_E, 5 * self.theta_E)
            ax1.set_ylim(-5 * self.theta_E, 5 * self.theta_E)

        plt.show()

    def plot_light_curves(self, model, bandpass, day_range, micro_day_range, add_microlensing):
        """
        Plots the apparent magnitudes of the individual light curves of the lensed supernova images, as well as their
        combined light curve. Black dots indicate the observation time stamps.

        :param model: SNcosmo model for the supernova light curve
        :param bandpass: LSST filter as input for SNcosmo
        :param day_range: array with a range of time steps to cover the lensed supernova evolution
        :param micro_day_range: array of shape [len(day_range), num_images] containing the microlensing contribution
               for each image and each time step
        :param add_microlensing: bool, if False: only compute macro light curves, if True: add microlensing contributions
        :return: plots the individual light curves, combined light curve, and observation time stamps
        """
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
        fig2.gca().invert_yaxis()
        ax2.set_xlabel("Day", fontsize=18)
        ax2.set_ylabel("Apparent magnitude", fontsize=18)
        ax2.set_title(r"Light curve", fontsize=25)

        colours = ['#3bb852', '#1fa3a3', '#205cd4', '#7143bf']
        fluxes = []

        # Plot the light curve for each image
        for image in range(len(self.td_images)):
            # Calculate the flux and append to total flux
            flux = model.bandflux(bandpass, time=day_range - self.td_images[image])
            flux *= self.macro_mag[image]
            fluxes.append(flux)
            # Calculate and plot the apparent magnitudes
            mags_range = model.bandmag(bandpass, time=day_range - self.td_images[image], magsys='ab')
            mags_range -= 2.5 * np.log10(self.macro_mag[image])
            if not add_microlensing:
                ax2.plot(day_range, mags_range, lw=2, label="Im " + str(image + 1) + " macro",
                         color=colours[image])
            # Calculate the apparent magnitudes with microlensing contributions
            elif add_microlensing:
                ax2.plot(day_range, mags_range, ls='--', lw=2, label="Im " + str(image + 1) + " macro",
                         color=colours[image], alpha=0.5)
                microlensing_lightcurve = mags_range + micro_day_range[:, image]
                ax2.plot(day_range, microlensing_lightcurve, lw=2, label="Im " + str(image + 1) + " microlensed",
                         color=colours[image])


        # Calculate the total flux of all images combined, convert to total apparent magnitude and plot
        zeropoint = sncosmo.ABMagSystem(name='ab').zpbandflux(bandpass)
        total_flux = np.sum(fluxes, axis=0)
        total_mag = -2.5 * np.log10(total_flux / zeropoint)
        total_lightcurve = interp1d(day_range, total_mag, bounds_error=False)
        # ax2.plot(day_range, total_mag, lw=4, color="gray", label="Total", zorder=1, alpha=0.5)
        # Plot observations
        # ax2.plot(self.days, total_lightcurve(self.days), '.', ms=10, color="black", label="Observations")
        for obs in range(len(self.days)):
            ax2.axvline(x=self.days[obs], color="black", lw=0.5, label="Observations" if obs == 0 else None, zorder=1)
        ax2.legend(loc=(1.01, 0.5), fontsize=18)

        plt.show()

    def plot_observations(self, time_series):
        """
        Plots the time evolution of the lensed supernova as difference images at the observation times.

        :param time_series: list of length [obs_upper_limit] containing simulated images of the lensed supernova at
           different time stamps corresponding to different observations
        :return: plots the lensed supernova difference images at the observation time stamps
        """
        fig3, ax3 = plt.subplots(3, 5, figsize=(15, 10))
        ax3 = ax3.flatten()
        fig3.suptitle("Observations", fontsize=25)

        for s in range(15):
            # Check if an observation exists
            if not scipy.sparse.issparse(time_series[s]):
                ax3[s].matshow(time_series[s], origin='lower', extent=[self.min_coordinate, self.max_coordinate,
                                                                       self.min_coordinate, self.max_coordinate])
                ax3[s].xaxis.set_ticks_position('bottom')
                ax3[s].set_title("t = " + str(int(np.around(self.days[s], 0))) + " days", fontsize=18)
            else:
                ax3[s].set_aspect('equal')

        plt.show()


def main():

    print("test")


if __name__ == '__main__':
    main()

