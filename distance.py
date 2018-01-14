# encoding : utf-8

import numpy as np
import os.path

import scipy.io.wavfile as wav
from scipy.linalg import toeplitz


def auto_covariance(s, delta):
    """
    Process auto-covariance for a signal with a specific time offset.

    :param s: [list of float] list of values for audio signal
    :param delta: [int] offset in number of samples for processing auto-covariance
    :return: [float] : auto-covariance processed
    """
    nb = len(s)
    s1 = s[0:nb - 1 - delta]
    s2 = s[delta:nb - 1]

    return np.mean(s1*s2)


def process_distance(mat_a, mat_b, wd, wv, wh):
    """
    Process distance matrix between 2 LPC matrices.

    :param mat_a: [numpy.array] LPC matrix of 1st signal
    :param mat_b: [numpy.array] LPC matrix of 2nd signal
    :param wd: [float] value for wd coefficient
    :param wv: [float] value for wv coefficient
    :param wh: [float] value for wh coefficient
    :return: [numpy.array] distance matrix processed, [float] distance processed
    """
    if mat_a.shape[0] == 0 or mat_b.shape[0] == 0:
        return None, 1000000

    mat_g = np.zeros((mat_a.shape[0], mat_b.shape[0]))

    for j in range(mat_g.shape[1]):  # for each column
        for i in range(mat_g.shape[0]):  # for each line
            d_ij = np.linalg.norm(mat_a[i] - mat_b[j])
            if i == 0 and j == 0:
                mat_g[0, 0] = d_ij
            elif j == 0:
                mat_g[i, j] = mat_g[i-1][j] + wv * d_ij
            elif i == 0:
                mat_g[i, j] = mat_g[i][j-1] + wh * d_ij
            else:
                mat_g[i, j] = min(mat_g[i - 1][j] + wv * d_ij,
                                  mat_g[i - 1][j - 1] + wd * d_ij,
                                  mat_g[i][j - 1] + wh * d_ij)

    return mat_g, mat_g[-1, -1] / (mat_g.shape[0] + mat_g.shape[1])


def construct_all_lpc_coefficients(input, output, n, window_length):
    """
    Construct and save LPC coefficients for each audioo signal.

    :param input: [str] directory name which contains set of .wav files
    :param output: [str] directory name in which LPC coefficients will be saved in .npy files
    :param n: [int] number of LPC coefficients for each time window
    :param window_length: [int] number of samples for each time window
    :return: None
    """
    files = os.listdir(input)
    for file in files:
        mat_a = process_lpc_coefficients(input + file, n, window_length)
        np.save(output + file[:-4], mat_a)


def process_lpc_coefficients(file, n, window_length):
    """
    Process LPC coefficients for a specific .wav file.

    :param file: [str] filename for which LPC coefficients will be processed
    :param n: [int] number of LPC coefficients for each time window
    :param window_length: [int] number of samples for each time window
    :return: [numpy.array] matrix of LPC coefficients processed
    """
    Fe, x = wav.read(file)

    # if signal x is stereo --> to mono
    if isinstance(x[0], np.ndarray):
        x = x.sum(axis=1) / 2

    sigma = np.zeros(n)
    sigma[0] = 1

    hamming_window = np.hamming(window_length)
    mat_a = []

    offset = 0
    while offset + window_length <= len(x):
        s = np.multiply(hamming_window, x[offset:offset + window_length])

        # LPC processing
        R_coeffs = [auto_covariance(s, delta) for delta in range(n)]
        R = toeplitz(R_coeffs)
        a = np.dot(np.linalg.inv(R), sigma)
        a = a / a[0]
        mat_a.append(a)
        offset += window_length // 2  # hamming window overlap (50%)
    return np.array(mat_a)