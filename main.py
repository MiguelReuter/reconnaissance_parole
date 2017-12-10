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


def init_parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--construct", help="construct and save LPC coefficients for all data", action="store_true")
    _parser.add_argument("--input_path", help="path for the input data", type=str, default='spoken_digit_dataset/')
    _parser.add_argument("--output_path", help="path for the output data", type=str, default='output/')

    _parser.add_argument("-n", help="number of LPC coefficients used", type=int, default=15)
    _parser.add_argument("-w", help="length of window in samples", type=int, default=240)
    _parser.add_argument("-s", help="prediction error (sigma²)", type=float, default=1.)
    return _parser


def init(_parser):
    _args = _parser.parse_args()

    _coeffs_nb = _args.n
    _window_length = _args.w
    _sigma_2 = _args.s
    _input_path = _args.input_path
    _output_path = _args.output_path

    # construct and save all lpc coefficients
    if _args.construct:
        construct_all_lpc_coefficients(_input_path, _output_path)

    return _coeffs_nb, _window_length, _sigma_2, _input_path, _output_path


def construct_all_lpc_coefficients(input, output):
    files = os.listdir(input)
    for file in files:
        mat_a = process_lpc_coefficients(input + file, coeffs_nb, sigma_2, window_length)
        np.save(output + file[:-4], mat_a)


def process_lpc_coefficients(file, coeffs_nb, sigma_2, window_length):
    Fe, x = wav.read(file)

    # if signal x is stereo --> to mono
    if isinstance(x[0], np.ndarray):
        x = x.sum(axis=1) / 2

    sigma = np.zeros(coeffs_nb)
    sigma[0] = sigma_2

    hamming_window = np.hamming(window_length)
    mat_a = []

    offset = 0
    while offset + window_length <= len(x):
        s = np.multiply(hamming_window, x[offset:offset + window_length])

        # LPC processing
        R_coeffs = [autocovariance(s, delta) for delta in range(coeffs_nb)]
        R = toeplitz(R_coeffs)
        a = np.dot(np.linalg.inv(R), sigma)

        mat_a.append(a)
        offset += window_length // 2  # hamming window overlap (50%)
    mat_a = np.array(mat_a)
    return mat_a


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    [coeffs_nb, window_length, sigma_2,
     input_path, output_path] = init(parser)

    # example of lpc coefficients loading
    lpc = np.load(output_path + "9_theo_9.npy")
    print(lpc.shape)