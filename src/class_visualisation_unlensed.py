#! /bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# plt.rc("font", family="serif")
# plt.rc("text", usetex=True)


class Visualisation_Unlensed:

    def __init__(self, data_class, obs_days, obs_days_filters):
        """
        This class visualises a lens system by printing out its properties and by plotting the time delay surface,
        light curves and observations.

        :param data_class: instance of ImageData() from Lenstronomy containing the image properties
        :param obs_days: array containing the time stamps (in days) of the observations
        :param obs_days_filters: array containing the filters corresponding to obs_days
        """

        self.obs_days = obs_days
        self.obs_days_filters = obs_days_filters
        self.data_class = data_class

    def print_properties(self, z_source, H_0, peak_brightness_image):
        """
        Prints out the key properties of the lensed supernova system.

        :param peak_brightness_image: array of length [num_filters, num_images] containing the brightest apparent magnitudes of each
               of the supernova images for each  band
        :param z_source: redshift of the supernova (float)
        :param H_0: value of the Hubble constant used for the background cosmology in this current lens configuration (float)
        :return: printed statements about the key properties of the lens system
        """
        print(" ")
        print("Supernova redshift: ", np.around(z_source, 2))
        print("Hubble constant: ", np.around(H_0, 2))
        print("Peak brightness images for r,i,z,y bands:")
        print(np.around(peak_brightness_image, 2))
        print(" ")

    def plot_light_curves(self, model, day_range, model_mag):
        """
        Plots the apparent magnitudes of the individual light curves of the lensed supernova images as seen from the
        observations in the different bands.

        :param model: SNcosmo model for the supernova light curve
        :param day_range: array with a range of time steps to cover the lensed supernova evolution
        :param obs_mag: array with observed apparent magnitudes of the supernova
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

        mags = model.bandmag('lssti', time=day_range, magsys='ab')
        ax2.plot(day_range, mags, color='black', alpha=0.5, lw=1.5, label=r"light curve in $i$-band")

        for obs in range(len(self.obs_days)):
            day = self.obs_days[obs]
            band = 'lsst' + self.obs_days_filters[obs]

            ax2.plot(day, model_mag[obs], color=colours[band], marker=markers[band], ms=10, label=band)

        legend_handles = [
            Line2D([0], [0], marker=markers['lsstr'], color=colours['lsstr'], label=r'lsst $r$', ms=10, lw=0),
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

        # plt.savefig("../results/figures/Lightcurves_unlensed.png", dpi=200, bbox_inches='tight')

        plt.show()

    def plot_light_curves_perband(self, model, day_range, model_mag, obs_mag, obs_mag_error):
        """
        Plots the apparent magnitudes of the individual light curves of the lensed supernova images as seen from the
        observations in the different bands.

        :param model: SNcosmo model for the supernova light curve
        :param day_range: array with a range of time steps to cover the lensed supernova evolution
        :param model_mag: array with ground truth apparent magnitudes
        :param obs_mag: array with observed (perturbed) apparent magnitudes
        :param obs_mag_error: array with magnitude errors
        :return: plots the individual light curves, combined light curve, and observation time stamps
        """

        fig2, ax2 = plt.subplots(4, 1, figsize=(7, 12))
        fig2.gca().invert_yaxis()
        fig2.suptitle(r"Observations in each band", fontsize=25)
        fig2.subplots_adjust(hspace=0.3)

        colours = {'lsstg': '#377eb8', 'lsstr': '#4daf4a',
                   'lssti': '#e3c530', 'lsstz': '#ff7f00', 'lssty': '#e41a1c'}

        markers = {'lsstg': 'v', 'lsstr': '^',
                   'lssti': '<', 'lsstz': 'o', 'lssty': 's'}

        bands = ['lsstr', 'lssti', 'lsstz', 'lssty']

        for b in range(4):

            band = bands[b]
            max_obs, min_obs = 24, 24
            max_lc, min_lc = [], []

            mags = model.bandmag(band, time=day_range, magsys='ab')
            max_lc.append(np.max(mags[np.isfinite(mags)]))
            min_lc.append(np.min(mags[np.isfinite(mags)]))

            ax2[b].plot(day_range, mags, color='black', alpha=0.5, lw=1.5)

            for obs in range(len(self.obs_days)):
                day = self.obs_days[obs]
                obs_band = 'lsst' + self.obs_days_filters[obs]

                if obs_band == band:
                    ax2[b].plot(day, obs_mag[obs], color=colours[band], marker=markers[band], ms=8, label=band)
                    ax2[b].plot(day, model_mag[obs], color='black', marker='.', ms=5)
                    ax2[b].vlines(day, obs_mag[obs] - obs_mag_error[obs], obs_mag[obs] + obs_mag_error[obs], color=colours[band])

                    if np.isfinite((obs_mag[obs])) and (obs_mag[obs]) < min_obs:
                        min_obs = (obs_mag[obs])
                    if np.isfinite((obs_mag[obs])) and (obs_mag[obs]) > max_obs:
                        max_obs = (obs_mag[obs])

            lim_max = max([max(max_lc) + 1, max_obs])
            lim_min = min([min(min_lc) - 1, min_obs])

            ax2[b].set_ylim(lim_max, lim_min)
            ax2[b].text(0.83, 0.057, str(bands[b]), transform=ax2[b].transAxes, fontsize=17, zorder=100)
            ax2[b].set_xlabel("Day", fontsize=18)
            ax2[b].set_ylabel("Apparent magnitude", fontsize=18)

            # plt.savefig("../results/figures/Lightcurves_weather_multiband.png", dpi=200, bbox_inches='tight')


def main():
    print("test")


if __name__ == '__main__':
    main()

