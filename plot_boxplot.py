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
        #'li6400xt_stomatal_conductance_mol/m2/s',
        'li6400xt_transpiration_rate_mmol/m2/s',
        #'li6400xt_vapour_pressure_deficit_kPa',
        #'li6400xt_leaf_temperature_C',
    ]

    bar_width = 0.275

    xoffsets = np.arange(len(data_sources))
    xoffsets = xoffsets - np.mean(xoffsets)
    xoffsets *= bar_width

    for n_clips in [1, 7]:
        fig_test, axs_test = plt.subplots(sharex=True, sharey=True)
        for idx_x, data_source, xoffset in zip(range(len(data_sources)), data_sources, xoffsets):
            df = dict()

            discard_interval = discard_intervals[0]

            sfn = os.path.join("./cache", f"size_{data_source}_ridge_{24 - discard_interval * 2}h_results.pkl")

            if not os.path.isfile(sfn):
                print(sfn, "not found")
                continue

            performance = pickle.load(open(sfn, 'rb'))

            tasks = deepcopy(plot_tasks)
            y = []

            for task in tasks:
                plot_key = max(performance[task].keys())
                plot_key = n_clips
                y, lt = [], []
                for j in range(len(performance[task][plot_key])):
                    y.append(performance[task][plot_key][j]["optimal_median"]["nmse"].test)
                    lt.append(performance[task][plot_key][j]["labels"])

                print(lt)

                df[f"{task}"] = y
            df["x"] = np.arange(len(y), dtype=int) + 1

            df = pd.DataFrame(df)
            df.to_csv(f"plot_data/box_plot_{data_source}_{n_clips}_clips_{24 - discard_interval * 2}h.csv", index=False)