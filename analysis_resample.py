#!/usr/bin/env python3
from copy import deepcopy

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import timedelta, datetime
import pickle
import numpy as np
import os
from itertools import combinations
from collections import namedtuple
import argparse
from typing import List
from analysis_ridge import train_ridge_task, get_ridge_split, get_tasks
from scipy import signal

from analysis_ridge import input_labels


def run_hpc_resample_script(run, tasks, discard_interval, resample_factors):
    # src_dir = "/user/gent/420/vsc42053/reservoir_sensing/experiments/nature_plants_analysis"
    src_dir = "."

    print(f"Running {run}")

    for use_calibration in [True]:
        sfn = os.path.join(src_dir, "cache")
        if not os.path.isdir(sfn):
            os.makedirs(os.path.join(sfn))
        if use_calibration:
            sfn = os.path.join(sfn, f"resample_{run}_ridge_{24 - discard_interval * 2}h_results.pkl")
        else:
            sfn = os.path.join(sfn, f"resample_nc_{run}_ridge_{24 - discard_interval * 2}h_results.pkl")

        if os.path.isfile(sfn):
            print(f"{sfn} already exists. Skipping.")
            performance_all = pickle.load(open(sfn, 'rb'))
        else:
            performance_all = dict()
        for decimation_factors in resample_factors:
            if isinstance(decimation_factors, int):
                decimation_factors = (decimation_factors,)
            rf = np.product(decimation_factors, dtype=int)
            print(f"Running with resample factor {rf}.")
            if use_calibration:
                print("Running with calibration")
            else:
                print("Running without calibration")

            if rf in performance_all:
                print("Skipping this value.")
                continue

            fn = os.path.join(src_dir, "data", f"{run}_full_data.csv")
            df = pd.read_csv(fn)

            print("Data loaded.")

            # resample data
            df_new = dict()

            all_labels = deepcopy(input_labels[run])
            all_labels += tasks
            for i in all_labels:
                x = df[i].values
                new_length = len(x) // rf
                x = x[:new_length*rf]
                for q in decimation_factors:
                    if q > 1:
                        x = signal.decimate(x, q=q, ftype="fir", zero_phase=True)
                df_new[i] = x

            print(f"Going from {len(df)} to {len(df_new[i])}.")

            df_new["time"] = df["time"].values[:new_length*rf:rf]
            df_new["train_val_test_split"] = df["train_val_test_split"].values[:new_length * rf:rf]
            df_new[f"di_{discard_interval}"] = df[f"di_{discard_interval}"].values[:new_length * rf:rf]

            discard_pre, discard_post = 3*60*60 // rf, 1*60*60 // rf
            for i in df_new:
                if discard_post > 0:
                    df_new[i] = df_new[i][discard_pre:-discard_post]
                else:
                    df_new[i] = df_new[i][discard_pre:]

            df = pd.DataFrame(df_new)
            print("New resampled dataframe constructed.")

            df_always_train, df_test, df_train_val = get_ridge_split(
                df,
                discard_interval=discard_interval)

            print("Preprocessing done.")

            # get correct labels
            labels = deepcopy(input_labels[run])
            if not use_calibration:
                labels = [i.replace("_um", "_nc_au") for i in labels]

            print("Training...")

            print(f"Input labels: {labels}")

            performance = dict()
            for task in tasks:
                print(f"Running {task}")
                performance = train_ridge_task(
                    df_always_train=df_always_train,
                    list_df_train_val=df_train_val,
                    df_test=df_test,
                    task=task,
                    input_labels=labels,
                    alphas=np.power(10.0, np.arange(-10, 10)),
                    size_analysis=7,
                    performance=performance)

            performance_all[rf] = performance

            print(performance_all.keys())

        print("\nStoring results.")

        pickle.dump(performance_all, open(sfn, 'wb'))


def main():
    discard_intervals = [2, 6, 9]
    runs = ["control", "strawberry_1", "strawberry_2"]
    tasks = [
        "li6400xt_external_probe_air_temperature_C",
        "li6400xt_external_probe_relative_humidity_percent",
        "li6400xt_photosynthetic_rate_umol/m2/s",
        "li6400xt_stomatal_conductance_mol/m2/s",
        "li6400xt_transpiration_rate_mmol/m2/s",
        "li6400xt_vapour_pressure_deficit_kPa",
        # "li6400xt_sample_cell_air_temperature_C",
        "li6400xt_leaf_temperature_C",
        # "li6400xt_ref_cell_CO2_conc_umol/mol",
        # "li6400xt_sample_cell_CO2_conc_umol/mol",
        # "li6400xt_ref_cell_H2O_conc_mmol/mol",
        # "li6400xt_sample_cell_H2O_conc_mmol/mol",
        # "li6400xt_ref_cell_relative_humidity_conc_percent",
        # "li6400xt_sample_cell_relative_humidity_conc_percent",
        # "li6400xt_PAR_inside_chamber_umol/m2/s",
        "li6400xt_PAR_outside_chamber_umol/m2/s",
        "li6400xt_air_pressure_kPa",
        "light_sensor_lux",
        # "soil_water_content_au",
        "air_temperature_C",
        "relative_humidity_percent"
    ]

    rs_factors = [1, 2, 3, 4, 5, 7, 10, (5, 3), (5, 4), (10, 3), (10, 4), (10, 5),
                  (10, 6), (10, 8), (10, 10), (10, 10, 2), (10, 10, 3), (10, 10, 4),
                  (10, 10, 5), (10, 10, 6), (10, 10, 8), (10, 10, 10), (10, 10, 10, 2),
                  (10, 10, 10, 3), (10, 10, 10, 4), (10, 10, 10, 5)]
    for di in discard_intervals:
        for run in runs:
            run_hpc_resample_script(run=run, tasks=tasks, discard_interval=di, resample_factors=rs_factors)

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("-r", "--run", help="data intput source", type=str, default="grass_1")
    p.add_argument("-di", "--discard_interval", type=int, default=2)

    args = p.parse_args()

    tasks = [
        "li6400xt_external_probe_air_temperature_C",
        "li6400xt_external_probe_relative_humidity_percent",
        "li6400xt_photosynthetic_rate_umol/m2/s",
        "li6400xt_stomatal_conductance_mol/m2/s",
        "li6400xt_transpiration_rate_mmol/m2/s",
        "li6400xt_vapour_pressure_deficit_kPa",
        # "li6400xt_sample_cell_air_temperature_C",
        "li6400xt_leaf_temperature_C",
        # "li6400xt_ref_cell_CO2_conc_umol/mol",
        # "li6400xt_sample_cell_CO2_conc_umol/mol",
        # "li6400xt_ref_cell_H2O_conc_mmol/mol",
        # "li6400xt_sample_cell_H2O_conc_mmol/mol",
        # "li6400xt_ref_cell_relative_humidity_conc_percent",
        # "li6400xt_sample_cell_relative_humidity_conc_percent",
        # "li6400xt_PAR_inside_chamber_umol/m2/s",
        "li6400xt_PAR_outside_chamber_umol/m2/s",
        "li6400xt_air_pressure_kPa",
        "light_sensor_lux",
        # "soil_water_content_au",
        "air_temperature_C",
        "relative_humidity_percent"
    ]

    rs_factors = [1, 2, 3, 4, 5, 7, 10, (5, 3), (5, 4), (10, 3), (10, 4), (10, 5),
                  (10, 6), (10, 8), (10, 10), (10, 10, 2), (10, 10, 3), (10, 10, 4),
                  (10, 10, 5), (10, 10, 6), (10, 10, 8), (10, 10, 10), (10, 10, 10, 2),
                  (10, 10, 10, 3), (10, 10, 10, 4), (10, 10, 10, 5)]

    run_hpc_resample_script(run=args.run, tasks=tasks, discard_interval=args.discard_interval, resample_factors=rs_factors)