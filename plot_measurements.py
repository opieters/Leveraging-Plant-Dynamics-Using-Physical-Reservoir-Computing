import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import pickle
from copy import deepcopy
from itertools import combinations
from collections import namedtuple

if __name__ == "__main__":
    data_sources = [
        #"control",
        "strawberry_1",
        #"strawberry_2",
    ]
    plot_vars = [
        "leaf_thickness_6_um",
        "li6400xt_PAR_outside_chamber_umol/m2/s",
        "air_temperature_C",
    ]
    ylabels = [
        "LT [μm]",
        "PAR [μmol/m2/s]",
        "Tair [°C]"
    ]

    for run in data_sources:
        fn = os.path.join(f"data/{run}_data.csv")
        df = pd.read_csv(fn)
        df.loc[:, "time"] = pd.to_datetime(df["time"])
        df.set_index("time", drop=True, inplace=True)

        fig, axs = plt.subplots(sharex=True, nrows=len(plot_vars), ncols=1)
        fig.subplots_adjust(hspace=0, wspace=0)

        axs[0].set_title(run)
        for i, ax in enumerate(axs):
            if "leaf_thickness" in plot_vars[i]:
                y = df[plot_vars[i]]
                y = (y - np.mean(y)) / np.std(y)
                y = 10*y + 125
                ax.plot(y)
            else:
                ax.plot(df[plot_vars[i]])
            ax.set_xlabel("time")
            ax.set_ylabel(ylabels[i])

        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        plt.show()
        fig.savefig(f"plots/measurements_{run}.png", facecolor='white', edgecolor='none')
