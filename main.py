# encoding : utf-8

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


if __name__ == '__main__':
    Fe, x = wav.read('spoken_digit_dataset/4_theo_5.wav')
    #plt.plot(x)
    #plt.show()

    window_length = 240  # nb of ech in a window
    offset = 0
    coeffs_nb = 15

    while offset + window_length <= len(x):
        s = x[offset:offset+window_length]
        s = np.hamming(window_length)

        # LPC processing
        R_coeffs = [autocovariance(x, delta) for delta in range(coeffs_nb)]
        R = toeplitz(R_coeffs)

        offset += window_length//2  # hamming window overlap (50%)

    print(R)