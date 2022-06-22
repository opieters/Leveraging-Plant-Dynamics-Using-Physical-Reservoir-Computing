import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import pickle
from copy import deepcopy
from itertools import combinations
from collections import namedtuple
from analysis_ridge import EvalPoint
from plot_ridge_analysis import short_name_map


if __name__ == "__main__":
    #plt.rcParams["figure.figsize"] = (20, 15)

    data_sources = [
        "control",
        "strawberry_1",
        "strawberry_2",
    ]

    discard_intervals = [4]

    plot_tasks = [
        #'li6400xt_photosynthetic_rate_umol/m2/s',
        #'li6400xt_stomatal_conductance_mol/m2/s',
        #'li6400xt_transpiration_rate_mmol/m2/s',
        #'li6400xt_vapour_pressure_deficit_kPa',
        #'li6400xt_leaf_temperature_C',
        'li6400xt_PAR_outside_chamber_umol/m2/s',
        #'air_temperature_C',
        #'relative_humidity_percent'
    ]

    df = dict()

    for task in plot_tasks:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

        discard_interval = 4
        for idx_x, data_source in enumerate(data_sources):

            sfn = os.path.join("./cache", f"nl_{data_source}_ridge_{24 - discard_interval * 2}h_results.pkl")

            if not os.path.isfile(sfn):
                print(sfn, "not found")
                continue

            performance = pickle.load(open(sfn, 'rb'))

            tasks = deepcopy(plot_tasks)
            x = list(performance.keys())
            y, yerr = [], []

            for delay in x:
                i = 7
                y.append(np.mean([performance[delay][task][i][j]["optimal_median"]["nmse"].test for j in
                                  range(len(performance[delay][task][i]))]))
                yerr.append(np.std([performance[delay][task][i][j]["optimal_median"]["nmse"].test for j in
                                    range(len(performance[delay][task][i]))]))

            df["degree"] = x
            df[f"y_{data_source}_{task}"] = y
            df[f"yerr_{data_source}_{task}"] = yerr

            ax.errorbar(x, y, yerr, fmt="-*", label=data_source)

        ax.set_title(task)
        ax.set_ylim([0, 1.1])
        ax.set_xlabel("polynomial degree")
        ax.set_ylabel("NMSE [-]")
        ax.grid()
        ax.legend()

        fig.savefig(f"plots/nl_plot_{task.replace('/', '_')}.png", facecolor='white', edgecolor='none')
        plt.show()

    df = pd.DataFrame(df)
    df.to_csv(f"plot_data/nl_effect_{24 - discard_interval * 2}h.csv", index=True, index_label="index")
