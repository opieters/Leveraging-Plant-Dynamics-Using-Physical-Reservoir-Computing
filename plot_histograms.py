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

    discard_intervals = [4]

    plot_tasks = [
        'li6400xt_PAR_outside_chamber_umol/m2/s',
        'air_temperature_C',
        'relative_humidity_percent',
        'li6400xt_photosynthetic_rate_umol/m2/s',
        'li6400xt_stomatal_conductance_mol/m2/s',
        'li6400xt_transpiration_rate_mmol/m2/s',
        'li6400xt_vapour_pressure_deficit_kPa',
        'li6400xt_leaf_temperature_C',
    ]

    bar_width = 0.275

    xoffsets = np.arange(len(data_sources))
    xoffsets = xoffsets - np.mean(xoffsets)
    xoffsets *= bar_width

    fig, ax = plt.subplots()

    fig_test, axs_test = plt.subplots(sharex=True, sharey=True)

    # write plot data to file
    df = dict()

    for n_clips in [1, 7]:
        for idx_x, data_source, xoffset in zip(range(len(data_sources)), data_sources, xoffsets):
            discard_interval = discard_intervals[0]

            sfn = os.path.join("./cache", f"size_{data_source}_ridge_{24 - discard_interval * 2}h_results.pkl")

            if not os.path.isfile(sfn):
                print(sfn, "not found")
                continue

            performance = pickle.load(open(sfn, 'rb'))

            tasks = deepcopy(plot_tasks)
            y, yerr = [], []

            for task in tasks:
                plot_key = max(performance[task].keys())
                plot_key = n_clips
                y.append(np.mean([performance[task][plot_key][j]["optimal_median"]["nmse"].test for j in
                                        range(len(performance[task][plot_key]))]))
                yerr.append(np.std([performance[task][plot_key][j]["optimal_median"]["nmse"].test for j in
                                          range(len(performance[task][plot_key]))]))

            df[f"y_{data_source}"] = y
            df[f"yerr_{data_source}"] = yerr
            df["x"] = tasks

            x = np.arange(len(y)) + xoffset
            axs_test.bar(x, y, bar_width, yerr=yerr, label=data_source)

            axs_test.set_ylim([0, 1.0])
            axs_test.set_ylabel("NMSE [-]")
            axs_test.grid()
            axs_test.set_xticks(np.arange(len(y)))
            axs_test.set_xticklabels([short_name_map[i] for i in tasks])

        df = pd.DataFrame(df)
        df.to_csv(f"plot_data/task_performance_{n_clips}_clips_{24 - discard_interval * 2}h.csv", index=True, index_label="index")

        axs_test.legend(ncol=1)
        fig_test.savefig(f"plots/histogram_plot_{n_clips}_clips_{24 - discard_interval * 2}h.png", facecolor='white', edgecolor='none')
        plt.show()