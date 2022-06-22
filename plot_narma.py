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

    discard_intervals = [2, 6, 9]
    discard_intervals = [4]

    tasks = [
        'li6400xt_PAR_outside_chamber_umol/m2/s',
        #"light_sensor_lux",
    ]

    narma_tasks = [f"NARMA{i}" for i in [2, 5, 10, 20, 50, 100]]

    bar_width = 0.25

    xoffsets = np.arange(len(narma_tasks))
    xoffsets = xoffsets - np.mean(xoffsets)
    xoffsets *= bar_width

    fig, axs = plt.subplots(nrows=1, ncols=len(tasks), sharex=True, squeeze=False)
    axs = axs[0]

    for ax, idx_x, task in zip(axs, range(len(axs)), tasks):
        # write plot data to file
        df = dict()

        for data_source, xoffset in zip(data_sources, xoffsets):
            discard_interval = discard_intervals[0]

            sfn = os.path.join("./cache", f"narma_{data_source}_ridge_{24 - discard_interval * 2}h_{task.replace('/','-')}_results.pkl")

            if not os.path.isfile(sfn):
                print(sfn, "not found")
                continue

            performance = pickle.load(open(sfn, 'rb'))

            y, yerr = [], []

            for narma_task in narma_tasks:
                plot_key = 7
                y.append(np.mean([performance[narma_task][plot_key][j]["optimal_median"]["nmse"].test for j in
                                        range(len(performance[narma_task][plot_key]))]))
                yerr.append(np.std([performance[narma_task][plot_key][j]["optimal_median"]["nmse"].test for j in
                                          range(len(performance[narma_task][plot_key]))]))

            df[f"y_{data_source}"] = y
            df[f"yerr_{data_source}"] = yerr
            df["x"] = narma_tasks
            df["n"] = [j.replace("NARMA", "") for j in narma_tasks]

            x = np.arange(len(y)) + xoffset
            print(x)
            ax.bar(x, y, bar_width, yerr=yerr, label=data_source)

            ax.set_ylim([0, 1.0])
            ax.set_ylabel("NMSE [-]")
            ax.grid()
            ax.set_xticks(np.arange(len(y)))
            ax.set_xticklabels(narma_tasks, rotation=90)
            ax.set_title(short_name_map[task])

        df = pd.DataFrame(df)
        df.to_csv(f"plot_data/narma_performance_{task.replace('/', '-')}_{24 - discard_interval * 2}h.csv", index=True, index_label="index")

    ax.legend(ncol=1)
    fig.savefig("plots/histogram_plot.png", facecolor='white', edgecolor='none')
    plt.show()