# encoding : utf-8


import argparse
import os.path

import scipy.io.wavfile as wav
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import numpy as np


def autocovariance(X, delta):
    N = len(X)
    Xs = np.average(X)
    auto_cov = 0
    times = 0
    for i in np.arange(0, N-delta):
        auto_cov += (X[i-delta]-Xs)*(X[i]-Xs)
        times += 1
    return auto_cov/times


def lpc_distance(a, b):
    # "foo" function
    if len(a) != len(b):
        return None

    d = sum([abs(a[i]/a[0] - b[i]/b[0]) for i in range(len(a))])
    return d


def process_mat_distance(mat_a, mat_b, wd, wv, wh):
    # elastic distance
    if mat_a.shape[1] != mat_b.shape[1]:
        return None

    mat_d = np.zeros((mat_a.shape[0], mat_b.shape[0]))
    # rajouter 0 au début de mat_a et mat_b ?
    for j in range(mat_d.shape[1]):  # for each column
        for i in range(mat_d.shape[0]):  # for each line
            d_ij = lpc_distance(mat_a[i], mat_b[j])

            g_im1_j = mat_d[i-1][j] if i > 0 else 0
            g_im1_jm1 = mat_d[i-1][j-1] if (i > 0 and j > 0) else 0
            g_i_jm1 = mat_d[i][j-1] if j > 0 else 0

            if j == 0:
                mat_d[i, j] = g_im1_j + wv * d_ij
            elif i == 0:
                mat_d[i, j] = g_i_jm1 + wh * d_ij
            else:
                mat_d[i, j] = min(g_im1_j + wv * d_ij,
                                  g_im1_jm1 + wd * d_ij,
                                  g_i_jm1 + wh * d_ij)

    return mat_d


def init_parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--construct", help="construct and save LPC coefficients for all data", action="store_true")
    _parser.add_argument("--input_path", help="path for the input data", type=str, default='spoken_digit_dataset/')
    _parser.add_argument("--output_path", help="path for the output data", type=str, default='output/')

    _parser.add_argument("-n", help="number of LPC coefficients used", type=int, default=15)
    _parser.add_argument("-w", help="length of window in samples", type=int, default=240)
    return _parser


def init(_parser):
    _args = _parser.parse_args()

    _coeffs_nb = _args.n
    _window_length = _args.w
    _input_path = _args.input_path
    _output_path = _args.output_path

    # construct and save all lpc coefficients
    if _args.construct:
        construct_all_lpc_coefficients(_input_path, _output_path, _coeffs_nb, _window_length)

    return _coeffs_nb, _window_length, _input_path, _output_path


def construct_all_lpc_coefficients(input, output, coeffs_nb, window_length):
    files = os.listdir(input)
    for file in files:
        mat_a = process_lpc_coefficients(input + file, coeffs_nb, window_length)
        np.save(output + file[:-4], mat_a)


def process_lpc_coefficients(file, coeffs_nb, window_length):
    Fe, x = wav.read(file)

    # if signal x is stereo --> to mono
    if isinstance(x[0], np.ndarray):
        x = x.sum(axis=1) / 2

    sigma = np.zeros(coeffs_nb)
    sigma[0] = 1

    hamming_window = np.hamming(window_length)
    mat_a = []

    offset = 0
    while offset + window_length <= len(x):
        s = np.multiply(hamming_window, x[offset:offset + window_length])

        # LPC processing
        R_coeffs = [autocovariance(s, delta) for delta in range(coeffs_nb)]
        R = toeplitz(R_coeffs)
        a = np.dot(np.linalg.inv(R), sigma)
        a = a / a[0]
        mat_a.append(a)
        offset += window_length // 2  # hamming window overlap (50%)
    mat_a = np.array(mat_a)
    return mat_a


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    [coeffs_nb, window_length,
     input_path, output_path] = init(parser)


    # example of lpc coefficients loading
    A_lpc = np.load(output_path + "9_jackson_9.npy")
    B_lpc = np.load(output_path + "9_jackson_45.npy")

    print(A_lpc.shape)
    print(B_lpc.shape)

    D = process_mat_distance(A_lpc, B_lpc, wd=1.0, wv=1.0, wh=1.0)

    plt.imshow(D)
    plt.colorbar(orientation='vertical')
    plt.show()
