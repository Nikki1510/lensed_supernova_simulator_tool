#! /bin/python3
import numpy as np
import time


class Timer:

    def __init__(self):
        """
        Class that times how long a code subroutine takes.
        __init__ creates a dictionary to save the durations of the different code subroutines.
        Meaning of timing names:

        'initiate' : Everything you have to do only once. Load data sets, start up dataframe, etc
        'general_properties' : Sample basic properties (H0, theta_E, redshifts, SN position)
        'lens_SN_properties': Sample lens and SN properties (lens and light curve model, calculate dt, magnification, etc)
        'detection_criteria_1': Check image multiplicity & magnification methods, based on peak brightness estimate
        'microlensing_1': Initiate microlensing class
        'microlensing_2': Calculate microlensing parameters kappa, gamma, s
        'microlensing_3': Load microlensing light curves
        'microlensing_4': Calculate microlensing contribution at peak
        'cadence': Generate cadence realisation and apply it to the lensed SN light curve
        'detection_criteria_2': Check image multiplicity & magnification methods, based on actual observations
        'finalise': Make last cuts to arrays, Show plots, save dataframe, update variables
        """

        self.timing_dict = {'initiate': [],
                            'general_properties': [],
                            'lens_SN_properties': [],
                            'detection_criteria_1': [],
                            'microlensing_1': [],
                            'microlensing_2': [],
                            'microlensing_3': [],
                            'microlensing_4': [],
                            'cadence': [],
                            'detection_criteria_2': [],
                            'finalise': []}

    def initiate(self, name):
        """
        To be used at the beginning of a code subroutine. This saves the starting time.
        :param name: name for the code subroutine (should match one of the names in the timing dictionary)
        :return: self.start records the starting time
        """

        self.start = time.time()
        self.name = name

    def end(self, name):
        """
        To be used at the end of a code subroutine. This calculates the duration and saves it in the timing dictionary.
        :param name: name for the code subroutine (should match one of the names in the timing dictionary)
        :return: updates self.timing_dict
        """

        end = time.time()
        duration = end - self.start

        try:
            self.timing_dict[name].append(duration)
        except:
            print("Make sure that ", name, " corresponds to one of the timing dictionary entries.")



def main():

    Timer()


if __name__ == '__main__':
    main()
