import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import pickle
from copy import deepcopy
from itertools import combinations
from collections import namedtuple

EvalPoint = namedtuple('EvalPoint', ['train', 'test'])

short_name_map = {
    "time": "time",
    "light": "light_sensor_lux",
    "SWC": "soil_water_content_au",
    "temp": "air_temperature_C",
    "rh": "relative_humidity_percent",
    "xTair": "li6400xt_external_probe_air_temperature_C",
    "xRH": "li6400xt_external_probe_relative_humidity_percent",
    "Cond": "li6400xt_stomatal_conductance_mol/m2/s",
    "Photo": "li6400xt_photosynthetic_rate_umol/m2/s",
    "Trmmol": "li6400xt_transpiration_rate_mmol/m2/s",
    "VpdL": "li6400xt_vapour_pressure_deficit_kPa",
    "Tair": "li6400xt_sample_cell_air_temperature_C",
    "Tleaf": "li6400xt_leaf_temperature_C",
    "CO2R": "li6400xt_ref_cell_CO2_conc_umol/mol",
    "CO2S": "li6400xt_sample_cell_CO2_conc_umol/mol",
    "H2OR": "li6400xt_ref_cell_H2O_conc_mmol/mol",
    "H2OS": "li6400xt_sample_cell_H2O_conc_mmol/mol",
    "RH_R": "li6400xt_ref_cell_relative_humidity_conc_percent",
    "RH_S": "li6400xt_sample_cell_relative_humidity_conc_percent",
    "PARi": "li6400xt_PAR_inside_chamber_umol/m2/s",
    "PARo": "li6400xt_PAR_outside_chamber_umol/m2/s",
    "Press": "li6400xt_air_pressure_kPa",
    "CTleaf": "li6400xt_computed_leaf_temperature_C",
    "Ci": "li6400xt_intercellular_CO2_conc_umol/ms/s"
}

short_name_map = {short_name_map[k]: k for k in short_name_map}


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (20, 15)

    data_sources = [
        "control",
        "strawberry_1",
        "strawberry_2",
    ]

    discard_intervals = [2, 4, 6, 9]

    plot_tasks = [
        'li6400xt_photosynthetic_rate_umol/m2/s',
        'li6400xt_stomatal_conductance_mol/m2/s',
        'li6400xt_transpiration_rate_mmol/m2/s',
        'li6400xt_vapour_pressure_deficit_kPa',
        'li6400xt_leaf_temperature_C',
        'li6400xt_PAR_outside_chamber_umol/m2/s',
        'air_temperature_C',
        'relative_humidity_percent'
    ]

    fig_test, axs_test = plt.subplots(len(discard_intervals), len(data_sources), sharex=True, sharey=True)
    # fig_train, axs_train = plt.subplots(len(discard_intervals), len(data_sources), sharex=True, sharey=True)
    for idx_x, data_source in enumerate(data_sources):
        for idx_y, discard_interval in enumerate(discard_intervals):

            sfn = os.path.join("./cache", f"size_{data_source}_ridge_{24 - discard_interval * 2}h_results.pkl")

            if not os.path.isfile(sfn):
                print(sfn, "not found")
                continue

            performance = pickle.load(open(sfn, 'rb'))

            # write plot data to file
            df = dict()

            tasks = deepcopy(plot_tasks)
            x = list(range(len(performance["air_temperature_C"])))
            y, yerr = dict(), dict()

            for task in tasks:
                y[task] = list(range(len(performance["air_temperature_C"])))
                yerr[task] = list(range(len(performance["air_temperature_C"])))
                for idx, i in enumerate(performance[task].keys()):
                    y[task][idx] = np.mean([performance[task][i][j]["optimal_median"]["nmse"].test for j in
                                            range(len(performance[task][i]))])
                    yerr[task][idx] = np.std([performance[task][i][j]["optimal_median"]["nmse"].test for j in
                                              range(len(performance[task][i]))])

                df[f"y_{task}"] = y[task]
                df[f"y_err_{task}"] = yerr[task]
            df["x"] = np.arange(len(y[task]), dtype=int) + 1

            for task in y:
                y[task] = np.array(y[task])
                yerr[task] = np.array(yerr[task])

            for task in y:
                axs_test[idx_y, idx_x].plot(x, y[task], label=short_name_map[task])
                # axs_test[idx_y, idx_x].fill_between(x, y[task] - yerr[task], y[task] + yerr[task], alpha=0.2)

            axs_test[idx_y, idx_x].set_ylim([0, 2])
            axs_test[idx_y, idx_x].set_title(f"TE {data_source} {24 - discard_interval * 2}h")
            axs_test[idx_y, idx_x].set_xlabel("number of sensors - 1")
            axs_test[idx_y, idx_x].set_ylabel("NMSE [-]")
            axs_test[idx_y, idx_x].grid()
            # axs_test[idx_y, idx_x].set_yscale('log')

            df = pd.DataFrame(df)
            df.to_csv(f"plot_data/size_effect_ridge_{data_source}_{24 - discard_interval * 2}h.csv", index=False)

    axs_test[0, 0].legend(ncol=2)
    # axs_train[0,0].legend(ncol=5)
    fig_test.savefig("plots/size_effect_ridge_with_calibration.png", facecolor='white', edgecolor='none')
    plt.show()