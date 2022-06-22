import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import pickle
from copy import deepcopy
from itertools import combinations
from collections import namedtuple

from plot_ridge_analysis import EvalPoint, short_name_map


if __name__ == "__main__":
    data_sources = [
        "control",
        "strawberry_1",
        "strawberry_2",
    ]

    plt.rcParams["figure.figsize"] = (26, 15)

    discard_intervals = [2, 6, 9]
    discard_intervals = [6]

    tasks = [
        'li6400xt_photosynthetic_rate_umol/m2/s',
        'li6400xt_stomatal_conductance_mol/m2/s',
        'li6400xt_transpiration_rate_mmol/m2/s',
        'li6400xt_vapour_pressure_deficit_kPa',
        'li6400xt_leaf_temperature_C',
        'li6400xt_PAR_outside_chamber_umol/m2/s',
        'air_temperature_C',
        'relative_humidity_percent'
    ]

    rs_factors = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 800, 1000, 2000, 3000, 4000, 5000]

    fig, axs = plt.subplots(nrows=1, ncols=len(tasks), sharex=True)

    for data_source in data_sources:
        discard_interval = discard_intervals[0]

        sfn = os.path.join("./cache", f"resample_{data_source}_ridge_{24 - discard_interval * 2}h_results.pkl")

        if not os.path.isfile(sfn):
            print(sfn, "not found")
            continue

        performance = pickle.load(open(sfn, 'rb'))

        for ax, idx_x, task in zip(axs, range(len(axs)), tasks):
            # write plot data to file
            df = dict()

            y, yerr = [], []

            for rf in rs_factors:
                plot_key = 7
                y.append(np.mean([performance[rf][task][plot_key][j]["optimal_median"]["nmse"].test for j in
                                        range(len(performance[rf][task][plot_key]))]))
                yerr.append(np.std([performance[rf][task][plot_key][j]["optimal_median"]["nmse"].test for j in
                                          range(len(performance[rf][task][plot_key]))]))

            df[f"y_{data_source}"] = y
            df[f"y_err_{data_source}"] = yerr
            df["x"] = rs_factors

            #x = np.log10(rs_factors)
            x = rs_factors
            ax.errorbar(x, y, yerr=yerr, label=data_source)

            ax.set_ylim([0, 1.0])
            ax.set_ylabel("NMSE [-]")
            ax.set_xlabel("sample int. [s]")
            ax.set_xscale('log')
            ax.grid()
            ax.set_title(short_name_map[task])

        #df = pd.DataFrame(df)
        #df.to_csv(f"plot_data/task_performance{data_source}_{24 - discard_interval * 2}h.csv", index=False)

    ax.legend(ncol=1)
    fig.savefig("plots/subsample_plot.png", facecolor='white', edgecolor='none')
    plt.show()