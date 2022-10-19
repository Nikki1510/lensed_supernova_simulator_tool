#! /bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
import sncosmo
import lenstronomy.Util.util as util
from lenstronomy.Util import constants as const
from matplotlib.lines import Line2D
# plt.rc("font", family="serif")
# plt.rc("text", usetex=True)

        
class Visualisation:

    def __init__(self, time_delay_distance, td_images, theta_E, data_class, macro_mag, obs_days, obs_days_filters):
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
        self.obs_days = obs_days
        self.obs_days_filters = obs_days_filters
        self.data_class = data_class

        self.max_coordinate, self.min_coordinate = max(self.data_class.pixel_coordinates[0][0]), \
                                                   min(self.data_class.pixel_coordinates[0][0])

    def print_properties(self, z_lens, z_source, H_0, micro_peak, peak_brightness_image):
        """
        Prints out the key properties of the lensed supernova system.

        :param peak_brightness_image: array of length [num_filters, num_images] containing the brightest apparent magnitudes of each
               of the supernova images for each  band
        :param z_lens: redshift of the lens galaxy (float)
        :param z_source: redshift of the supernova (float)
        :param H_0: value of the Hubble constant used for the background cosmology in this current lens configuration (float)
        :param micro_peak: array of length [num_images] containing the microlensing contributions at light curve peak
        :return: printed statements about the key properties of the lens system
        """
        print(" ")
        print("Lens redshift: ", np.around(z_lens, 2))
        print("Supernova redshift: ", np.around(z_source, 2))
        print("Einstein radius: ",np.around(self.theta_E, 2))
        print("Time delays: ", np.around(self.td_images, 2), "days")
        print("Macro magnification: ", np.around(self.macro_mag, 2))
        print("Microlensing contribution at peak (magnitudes): ", np.around(micro_peak, 2))
        print("Hubble constant: ", np.around(H_0, 2))
        print("Time-delay distance: ", np.around(self.time_delay_distance, 2))
        print("Peak brightness images for r,i,z,y bands:")
        print(np.around(peak_brightness_image, 2))
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

    def plot_light_curves(self, model, day_range, micro_day_range, add_microlensing):
        """
        Plots the apparent magnitudes of the individual light curves of the lensed supernova images as seen from the
        observations in the different bands.

        :param model: SNcosmo model for the supernova light curve
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
        # ax2.set_title(r"Light curve", fontsize=25)

        colours = {'lsstg': '#377eb8', 'lsstr': '#4daf4a',
                   'lssti': '#e3c530', 'lsstz': '#ff7f00', 'lssty': '#e41a1c'}

        markers = {'lsstg': 'v', 'lsstr': '^',
                   'lssti': '<', 'lsstz': 'o', 'lssty': 's'}

        def total_mag(model, day_range, day, band, td_images, macro_mag):
            """
            This function returns the combined magnitude of all SN images at a certain point in time (day)
            """

            fluxes = []
            for im in range(len(td_images)):
                flux = model.bandflux(band, time=day_range - td_images[im])
                flux *= macro_mag[im]
                fluxes.append(flux)

            zeropoint = sncosmo.ABMagSystem(name='ab').zpbandflux(band)
            total_flux = np.sum(fluxes, axis=0)
            total_mag = -2.5 * np.log10(total_flux / zeropoint)
            total_lightcurve = interp1d(day_range, total_mag, bounds_error=False)

            return total_lightcurve(day)

        for im in range(len(self.td_images)):
            mags = model.bandmag('lssti', time=day_range - self.td_images[im], magsys='ab')
            mags -= 2.5 * np.log10(self.macro_mag[im])
            ax2.plot(day_range, mags, color='black', alpha=0.5, lw=1.5, label=r"light curves in $i$-band" if im ==0 else None)

        for obs in range(len(self.obs_days)):
            day = self.obs_days[obs]
            band = 'lsst' + self.obs_days_filters[obs]

            ax2.plot(day, total_mag(model, day_range, day, band, self.td_images, self.macro_mag),
                     color=colours[band], marker=markers[band], ms=10, label=band)

        legend_handles = [Line2D([0], [0], marker=markers['lsstr'], color=colours['lsstr'], label=r'lsst $r$', ms=10, lw=0),
                          Line2D([0], [0], marker=markers['lssti'], color=colours['lssti'], label=r'lsst $i$', ms=10, lw=0),
                          Line2D([0], [0], marker='o', color='white'),
                          Line2D([0], [0], marker=markers['lsstz'], color=colours['lsstz'], label=r'lsst $z$', ms=10, lw=0),
                          Line2D([0], [0], marker=markers['lssty'], color=colours['lssty'], label=r'lsst $y$', ms=10, lw=0)]

        legend1 = plt.legend(handles=legend_handles, loc='lower right', ncol=2, handletextpad=.3, borderaxespad=.3,
                    labelspacing=.3, borderpad=.3, columnspacing=.4, fontsize=18)

        legend_handles2 = [Line2D([0], [0], color='black', lw=1.5, alpha=0.5, label=' ')]

        legend2 = plt.legend(handles=legend_handles2, loc=(0.75, 0.01), handletextpad=.3, borderaxespad=.3,
                   labelspacing=.3, borderpad=.3, columnspacing=.4, fontsize=18, frameon=False)

        ax2.add_artist(legend1)
        ax2.add_artist(legend2)

        ax2.text(0.83, 0.057, r'light curves ($i$)', transform=ax2.transAxes, fontsize=17, zorder=100)

        # plt.savefig("../results/figures/Lightcurves_multiband.png", dpi=200, bbox_inches='tight')

        """


    
            mag = model.bandmag(filter, time=day - self.td_images[image], magsys='ab')

            ax2.plot(day, mag)






        # Plot the light curve for each image
        for image in range(len(self.td_images)):
            # Calculate the flux and append to total flux
            flux = model.bandflux('lssti', time=day_range - self.td_images[image])
            flux *= self.macro_mag[image]
            fluxes.append(flux)
            # Calculate and plot the apparent magnitudes
            mags_range = model.bandmag('lssti', time=day_range - self.td_images[image], magsys='ab')
            mags_range -= 2.5 * np.log10(self.macro_mag[image])
            if not add_microlensing:
                ax2.plot(day_range, mags_range, lw=4, label="Im " + str(image + 1) + " macro",
                         color=colours[image])
            # Calculate the apparent magnitudes with microlensing contributions
            elif add_microlensing:
                ax2.plot(day_range, mags_range, ls='--', lw=2, label="Im " + str(image + 1) + " macro",
                         color=colours[image], alpha=0.5)
                microlensing_lightcurve = mags_range + micro_day_range[:, image]
                ax2.plot(day_range, microlensing_lightcurve, lw=2, label="Im " + str(image + 1) + " microlensed",
                         color=colours[image])


        # Calculate the total flux of all images combined, convert to total apparent magnitude and plot
        # zeropoint = sncosmo.ABMagSystem(name='ab').zpbandflux(bandpass)
        # total_flux = np.sum(fluxes, axis=0)
        # total_mag = -2.5 * np.log10(total_flux / zeropoint)
        # total_lightcurve = interp1d(day_range, total_mag, bounds_error=False)
        # ax2.plot(day_range, total_mag, lw=4, color="gray", label="Total", zorder=1, alpha=0.5)
        # Plot observations
        # ax2.plot(self.obs_days, total_lightcurve(self.obs_days), '.', ms=10, color="black", label="Observations")
        for obs in range(len(self.obs_days)):
            if self.obs_days_filters[obs] == 'u':
                obs_color = '#984ea3'
                ax2.text(1.2, 0.8, 'u-band', color=obs_color, transform = ax2.transAxes)
            elif self.obs_days_filters[obs] == 'g':
                obs_color = '#377eb8'
                ax2.text(1.05, 0.8, 'g-band', color=obs_color, transform=ax2.transAxes, fontsize=18)
            elif self.obs_days_filters[obs] == 'r':
                obs_color = '#4daf4a'
                ax2.text(1.05, 0.7, 'r-band', color=obs_color, transform=ax2.transAxes, fontsize=18)
            elif self.obs_days_filters[obs] == 'i':
                obs_color = '#e3c530'
                ax2.text(1.05, 0.6, 'i-band', color=obs_color, transform=ax2.transAxes, fontsize=18)
            elif self.obs_days_filters[obs] == 'z':
                obs_color = '#ff7f00'
                ax2.text(1.05, 0.5, 'z-band', color=obs_color, transform=ax2.transAxes, fontsize=18)
            elif self.obs_days_filters[obs] == 'y':
                obs_color = '#e41a1c'
                ax2.text(1.05, 0.4, 'y-band', color=obs_color, transform=ax2.transAxes, fontsize=18)
            ax2.axvline(x=self.obs_days[obs], color=obs_color, lw=1, label="Observations" if obs == 0 else None, zorder=1)
        # ax2.legend(loc=(1.01, 0.5), fontsize=18)


        # labels = {'o': 'g', 'o': 'r', 'o': 'i', 'o': 'z'}
        # ax2.legend(ncol=2, handletextpad=.3, borderaxespad=.3,
        #            labelspacing=.2, borderpad=.3, columnspacing=.4)

        # plt.savefig("../results/figures/Lightcurve_single_for_BOOM.pdf", transparent=True, bbox_inches='tight')
        """

        plt.show()

    def plot_observations(self, time_series):
        """
        Plots the time evolution of the lensed supernova as difference images at the observation times.

        :param time_series: list of length [obs_upper_limit] containing simulated images of the lensed supernova at
           different time stamps corresponding to different observations
        :return: plots the lensed supernova difference images at the observation time stamps
        """

        observations = 0

        for obs in range(40):
            if not scipy.sparse.issparse(time_series[obs]):
                observations += 1

        number = int(np.ceil(observations / 5))

        if number == 2:
            margin = 7; wspace = 0.1; hspace = 0.1
        elif number == 3:
            margin = 10; wspace = 0.18; hspace = 0.2
        elif number == 4:
            margin = 14; wspace = 0.16; hspace = 0.25
        elif number == 5:
            margin = 17; wspace = 0.14; hspace = 0.3
        elif number == 6:
            margin = 19; wspace = 0.1; hspace = 0.35
        elif number == 7:
            margin = 22; wspace = 0.01; hspace = 0.4
        elif number == 8:
            margin = 23; wspace = 0.001; hspace = 0.5
        else:
            margin = 7; wspace = 0.1; hspace = 0.1; number = 1

        fig3, ax3 = plt.subplots(number, 5, figsize=(15, margin))
        ax3 = ax3.flatten()
        fig3.suptitle("Observations", fontsize=25)
        fig3.subplots_adjust(wspace=wspace, hspace=hspace)

        for s in range(int(number*5)):
            # Check if an observation exists
            if not scipy.sparse.issparse(time_series[s]):
                ax3[s].matshow(time_series[s], origin='lower', extent=[self.min_coordinate, self.max_coordinate,
                                                                       self.min_coordinate, self.max_coordinate])
                ax3[s].xaxis.set_ticks_position('bottom')
                ax3[s].set_title("t = " + str(int(np.around(self.obs_days[s], 0))) + " days", fontsize=18)
                ax3[s].text(-2, -4, "filter = " + str(self.obs_days_filters[s]), color="white", fontsize=18)
            else:
                ax3[s].set_aspect('equal')

        plt.show()


def main():

    print("test")


if __name__ == '__main__':
    main()

