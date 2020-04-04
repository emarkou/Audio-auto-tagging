import numpy as np
import scipy.fftpack
import utils as utils

def choose_frequencies():
    """
    # provide three frequencies in a range between 1 and 50    
    :return: [int, int, int]
    """
    # *** TODO provide three frequencies between 1 and 50
    freq1 = 10
    freq2 = 25
    freq3 = 40
    # end TODO

    return [freq1, freq2, freq3]


def add_the_waves(freqs):
    """
    create three sinusoidal waves and one wave that is the sum of the three
    :param freqs: [int, int, int]
    :return: [np.array, np.array, np.array, np.array]
        representing wave1, wave2, wave3, sum of waves
        each array contains 500(by default) discrete values for plotting a sinusoidal
    """
    _, _, t = utils.get_wave_timing()
    w1, w2, w3 = utils.make_waves(t, freqs)

    # TODO sum the waves together to form sum_waves
    sum_waves = w1+w2+w3
    # end TODO

    return [w1, w2, w3, sum_waves]


def demo_fft(sum_waves):
    num_samples, spacing, _ = utils.get_wave_timing()

    # TODO create a Fast Fourier Transform of the waveform using scipy.fftpack.fft
    # named 'y_fft'
    y_fft = scipy.fftpack.fft(sum_waves, num_samples)
    # end TODO

    x_fft = np.linspace(0.0, 1.0/spacing, num_samples)
    return x_fft, y_fft
