# encoding : utf-8


import argparse
import matplotlib.pyplot as plt

from classification import *
from distance import *


def init_parser():
    """
    Create parser with specific arguments.

    :return: parser
    """
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--construct", help="construct and save LPC coefficients for all data", action="store_true")
    _parser.add_argument("--show_signals", help="show representation of signals specified by --file_1 and --file_2",
                         action="store_true")
    _parser.add_argument("--show_distance", help="show representation of distance matrix specified by "
                                                 "--file_1 and --file_2", action="store_true")
    _parser.add_argument("--classify", help="classify data", action="store_true")
    _parser.add_argument("--input_path", help="path for the input data (used in --construct)", type=str,
                         default='spoken_digit_dataset/')
    _parser.add_argument("--output_path", help="path for the output data (used in --construct)", type=str,
                         default='output/')
    _parser.add_argument("--file_1", help="path for first file (for --show_signals and --show_distance)", type=str,
                         default='spoken_digit_dataset/0_theo_0.wav')
    _parser.add_argument("--file_2", help="path for second file (for --show_signals and --show_distance)", type=str,
                         default='spoken_digit_dataset/0_theo_1.wav')
    _parser.add_argument("-n", help="number of LPC coefficients used", type=int, default=15)
    _parser.add_argument("-w", help="length of window in samples", type=int, default=240)
    _parser.add_argument("--wh", help="value for w_h coefficient", type=float, default=1.0)
    _parser.add_argument("--wd", help="value for w_d coefficient", type=float, default=1.0)
    _parser.add_argument("--wv", help="value for w_v coefficient", type=float, default=1.0)
    _parser.add_argument("-k", help="specify number of nearest neighbors used to classify", type=int, default=5)

    return _parser


def show_signals(file_1, file_2):
    """
    Display the waveforms for file_1 and file_2 in a window.

    :param file_1: [str] filename for the first audio file
    :param file_2: [str] filename for the second audio file
    :return: None
    """
    fe_1, s1 = wav.read(file_1)
    fe_2, s2 = wav.read(file_2)

    # if signal is stereo --> to mono
    if isinstance(s1[0], np.ndarray):
        s1 = s1.sum(axis=1) / 2

    if isinstance(s2[0], np.ndarray):
        s2 = s2.sum(axis=1) / 2

    plt.plot([i/fe_1 for i in range(len(s1))], s1, label=file_1)
    plt.plot([i/fe_2 for i in range(len(s2))], s2, label=file_2)
    plt.title("Signals representation")
    plt.xlabel("time [s]")
    plt.ylabel("value")
    plt.legend(bbox_to_anchor=(0., .92, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def show_distance(file_1, file_2, n, w, wd, wv, wh):
    """
    Display the distance matrice between audio file file_1 and file_2.

    :param file_1: [str] filename for the first audio file
    :param file_2: [str] filename for the second audio file
    :param n: [int] number of LPC coefficients for each time window
    :param w: [int] number of samples for each time window
    :param wd: [float] value for wd coefficient
    :param wv: [float] value for wv coefficient
    :param wh: [float] value for wh coefficient
    :return: None
    """
    lpc_1 = process_lpc_coefficients(file_1, n, w)
    lpc_2 = process_lpc_coefficients(file_2, n, w)
    d, _ = process_distance(lpc_1, lpc_2, wd=wd, wv=wv, wh=wh)

    plt.imshow(d)
    plt.title("Distance matrix")
    plt.xlabel(file_1)
    plt.ylabel(file_2)
    plt.colorbar(orientation='vertical')
    plt.show()


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    if args.construct:
        construct_all_lpc_coefficients(args.input_path, args.output_path, args.n, args.w)

    if args.show_distance:
        show_distance(args.file_1, args.file_2, args.n, args.w, args.wd, args.wv, args.wh)

    if args.show_signals:
        show_signals(args.file_1, args.file_2)

    if args.classify:
        results = classify(k=args.k, wd=args.wd, wv=args.wv, wh=args.wh)
        classifier_precision = results.count(True) / len(results)
        print("--------------------------------------------")
        print("Classifier precision : ", classifier_precision)

        with open("log.txt", "a") as log_file:
            log_file.write("--------------------------------------------\n")
            log_file.write("Classifier precision : " + str(classifier_precision) + "\n")