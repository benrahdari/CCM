import numpy as np
import itertools
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import math

mpl.rcParams["figure.figsize"] = [6, 4]
mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["errorbar.capsize"] = 3
mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["grid.linewidth"] = 0.75
mpl.rcParams["lines.linewidth"] = 0.75
mpl.rcParams["patch.linewidth"] = 0.75
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = "medium"

import platform
print("python %s" % platform.python_version())
print("matplotlib %s" % mpl.__version__)

def linestyle2dashes(style):
    if style == "--":
        return (3, 3)
    elif style == ":":
        return (0.5, 2.5)
    else:
        return (None, None)

D = np.loadtxt("data/CarouselStudy.csv", delimiter=",").astype(int)
D[D[:, 0] == 8, 0] = 3

study_id = D[:, 0]
chosen_row = D[:, 1]
chosen_col = D[:, 2]

def fit_total_variation(M, probs, model="tcm"):
    best_params = ()
    best_dist = np.Inf
    best_P = np.zeros((6, 5))
    
    for pa in probs:
        for pa_discount in 1 - probs:
            for pq in probs:
                i = np.outer(np.arange(6), np.ones(5))
                j = np.outer(np.ones(6), np.arange(5))
                k = 5 * i + j
                if model == "tcm":
                    pad = pa * np.power(pa_discount, k)  # position-adjusted attraction
                    P = np.exp(k * np.log(1 - pq) + k * np.log(1 - pad) + np.log(pad))  # click probabilities
                elif model == "ccm":
                    pad = pa * np.power(pa_discount, i + j)  # position-adjusted attraction
                    P = np.exp((i + j) * np.log(1 - pq) + k * np.log(1 - pad) + np.log(pad))  # click probabilities
                dist = np.abs(M - P).sum() / 2  # total variation distance
                if dist < best_dist:
                    best_params = (pa, pa_discount, pq)
                    best_dist = dist
                    best_P = P
    return best_params, best_dist, best_P
