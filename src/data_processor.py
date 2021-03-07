import numpy as np


def average_eeg_over_participants(eeg_representations):
    return np.average(eeg_representations, axis=0)
