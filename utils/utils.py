import matplotlib.pyplot as plt
import numpy as np

def sinusoid(freq):
    """
    return a sinusoidal of random amplitude and phase for a given frequency
    :param freq: 
    :return: 
    """
    phase = np.random.random()
    amplitude = 2 * (np.random.random_integers(1, 10))
    return amplitude * np.cos(2 * np.pi * freq - phase)


def get_wave_timing(num_samples=500, range_of_time = 5.0):
    """
    provide an array of time values of size num_samples spread evenly over range_of_time
    :param num_samples: int 
    :param range_of_time: float
    :return: int, float, np.array
    """
    # sample spacing
    spacing = range_of_time / num_samples
    # array for time samples
    t = np.linspace(0.0, range_of_time, num_samples)
    return num_samples, spacing, t


def make_waves(t, freqs):
    """
    convert three frequencies into arrays of discrete values representing sinusoidal waves
    :param freqs: [float, float, float]
    :return: [np.array, np.array, np.array]
    """
    w0 = sinusoid(t * freqs[0])
    w1 = sinusoid(t * freqs[1])
    w2 = sinusoid(t * freqs[2])
    return w0, w1, w2


# def display_sinusoids(time_array, f1, f2, f3, sum):
#     # plot three frequencies with random phase shifts on y axis
#     # plt.figure()
#     fig, ax = plt.subplots(4, 1)
#     # plt.subplot(411)  # 3 rows, 1 column, fignum 1
#     ax[0].plot(time_array, f1)
#     ax[0].set_title('1st frequency component')

#     # plt.subplot(412)  # 3 rows, 1 column, fignum 2
#     ax[1].plot(time_array, f2)
#     ax[1].set_title('2nd frequency component')

#     # plt.subplot(413)  # 3 rows, 1 column, fignum 3
#     ax[2].plot(time_array, f3)
#     ax[2].set_title('3rd frequency component')

#     # sum
#     # plt.subplot(414)  # 3 rows, 1 column, fignum 4
#     ax[3].plot(time_array, sum, 'r')
#     ax[3].set_title('Sum of components')
#     ax[3].set_ylabel('amplitude')
#     ax[3].set_xlabel('time')

#     # adjust format of display to make room for titles
#     plt.subplots_adjust(
#         top=0.94,
#         bottom=0.11,
#         left=0.11,
#         right=0.97,
#         hspace=0.65,
#         wspace=0.2
#     )
#     # plt.show()
#     return fig


# def display_fft(xf, yf):
#     num_samples = np.shape(yf)[0]
#     fig, ax = plt.subplots()
#     ax.plot(xf, 2.0 / num_samples * np.abs(yf[:num_samples // 2]))
#     plt.title('Fast Fourier Transform')
#     plt.xlabel('frequency')
#     plt.ylabel('amplitude')
#     return fig

