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

EvalPoint = namedtuple('EvalPoint', ['train', 'test'])

input_labels = {
    "control": [
        "leaf_thickness_1_um",
        "leaf_thickness_2_um",
        "leaf_thickness_3_um",
        "leaf_thickness_4_um",
        "leaf_thickness_5_um",
        "leaf_thickness_6_um",
        "leaf_thickness_7_um",
        "leaf_thickness_8_um"
    ],
    "strawberry_1": [
        "leaf_thickness_1_um",
        "leaf_thickness_2_um",
        "leaf_thickness_3_um",
        "leaf_thickness_4_um",
        "leaf_thickness_5_um",
        "leaf_thickness_6_um",
        "leaf_thickness_7_um",
        "leaf_thickness_8_um"
    ],
    "strawberry_2": [
        "leaf_thickness_1_um",
        "leaf_thickness_2_um",
        "leaf_thickness_3_um",
        "leaf_thickness_4_um",
        "leaf_thickness_5_um",
        "leaf_thickness_6_um",
        "leaf_thickness_7_um",
        "leaf_thickness_8_um"
    ],
}


def mse(x, y):
    return np.mean(np.power(x - y, 2.0))


def nmse(yh, y):
    return mse(yh, y) / mse(y, np.mean(y))


def get_tasks(df: pd.DataFrame, kind: str) -> List[str]:
    licor_variables = [
        "li6400xt_external_probe_air_temperature_C",
        "li6400xt_external_probe_relative_humidity_percent",
        "li6400xt_photosynthetic_rate_umol/m2/s",
        "li6400xt_stomatal_conductance_mol/m2/s",
        "li6400xt_intercellular_CO2_conc_umol/ms/s",
        "li6400xt_transpiration_rate_mmol/m2/s",
        "li6400xt_vapour_pressure_deficit_kPa",
        "li6400xt_computed_leaf_temperature_C",
        "li6400xt_sample_cell_air_temperature_C",
        "li6400xt_leaf_temperature_C",
        "li6400xt_ref_cell_CO2_conc_umol/mol",
        "li6400xt_sample_cell_CO2_conc_umol/mol",
        "li6400xt_ref_cell_H2O_conc_mmol/mol",
        "li6400xt_sample_cell_H2O_conc_mmol/mol",
        "li6400xt_ref_cell_relative_humidity_conc_percent",
        "li6400xt_sample_cell_relative_humidity_conc_percent",
        "li6400xt_PAR_inside_chamber_umol/m2/s",
        "li6400xt_PAR_outside_chamber_umol/m2/s",
        "li6400xt_air_pressure_kPa",
    ]
    environment = [
        "light_sensor_lux",
        "soil_water_content_au",
        "air_temperature_C",
        "relative_humidity_percent"
    ]
    if kind == "all":
        variables = licor_variables + environment
    elif kind == "licor":
        variables = licor_variables
    elif kind == "environment":
        variables = environment
    else:
        raise ValueError(f"Unknown target category {kind}")
    return [i for i in variables if i in df]


def get_ridge_split(df: pd.DataFrame, discard_interval=2):
    discard_mask = ~df[f"di_{discard_interval}"]
    data_split = df["train_val_test_split"]
    always_train_mask = (data_split == 1) & discard_mask
    test_mask = (data_split == -1) & discard_mask
    df_trains = []
    for i in range(2, np.max(data_split)+1):
        mask = data_split == i
        mask &= discard_mask
        df_trains.append(df[mask])

    # create new dataframes for train and test split
    return df[always_train_mask], df[test_mask], df_trains


def cv_ridge_train_model(df_always_train, list_df_train_val, df_test, task, ilabels, alphas, optimal_key="optimal",
                         performance_evaluation=dict()):
    # store input labels
    performance_evaluation["labels"] = ilabels

    for alpha in alphas:
        alpha_label = f"{alpha:.2e}"
        print(f"Running alpha={alpha:.2e}")

        if alpha_label in performance_evaluation:
            print(f"Skipping {alpha_label}, already in evaluation.")
            continue

        performance_evaluation[alpha_label] = dict()
        performance_evaluation[alpha_label]["cv_train_nmse"] = []
        performance_evaluation[alpha_label]["cv_val_nmse"] = []

        for val_index in range(len(list_df_train_val)):
            df_val = list_df_train_val[val_index]
            df_train = [df_always_train] + [df for idx, df in enumerate(list_df_train_val) if idx != val_index]
            df_train = pd.concat(df_train)

            X_train = np.stack([df_train[i].values for i in ilabels])
            X_train = X_train.transpose()

            X_val = np.stack([df_val[i].values for i in ilabels])
            X_val = X_val.transpose()

            y_train = df_train[task].values
            y_val = df_val[task].values

            model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha, solver="lsqr"))])
            model.fit(X_train, y_train)

            yhat_train = model.predict(X_train)
            yhat_val = model.predict(X_val)

            performance_evaluation[alpha_label]["cv_train_nmse"].append(nmse(yh=yhat_train, y=y_train))
            performance_evaluation[alpha_label]["cv_val_nmse"].append(nmse(yh=yhat_val, y=y_val))

    # select best alpha value and train test set
    for eval_fn, label in [(np.mean, 'mean'), (np.median, 'median')]:
        opt = f"{optimal_key}_{label}"
        # if opt in performance_evaluation:
        #    print("Skipping, {opt}")
        #    continue
        nmse_val_mean = dict()
        for alpha in alphas:
            alpha_label = f"{alpha:.2e}"
            nmse_val_mean[alpha] = eval_fn(performance_evaluation[alpha_label]["cv_val_nmse"])

        alpha_opt = min(nmse_val_mean, key=nmse_val_mean.get)

        df_train = [df_always_train] + list_df_train_val
        df_train = pd.concat(df_train)

        X_train = np.stack([df_train[i].values for i in ilabels])
        X_train = X_train.transpose()
        y_train = df_train[task].values

        model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha_opt, solver="sparse_cg"))])
        model.fit(X_train, y_train)

        X_test = np.stack([df_test[i].values for i in ilabels])
        X_test = X_test.transpose()
        y_test = df_test[task].values

        yhat_train = model.predict(X_train)
        yhat_test = model.predict(X_test)

        performance_evaluation[opt] = {
            f"alpha": alpha_opt,
            f"model": model,
            f"nmse": EvalPoint(test=nmse(yh=yhat_test, y=y_test), train=nmse(yh=yhat_train, y=y_train))
        }

    return performance_evaluation


def train_ridge_task(df_always_train, list_df_train_val, df_test, task, input_labels, alphas, size_analysis=None,
                      performance=dict()):
    print(f"Training {task}")
    if task not in performance:
        performance[task] = dict()

    if size_analysis is None:
        size_analysis_range = range(1, len(input_labels) + 1)
    else:
        if isinstance(size_analysis, list):
            size_analysis_range = size_analysis
        else:
            size_analysis_range = [size_analysis]

    for i in size_analysis_range:
        print(f"Training with {i} input features.")
        if i not in performance[task]:
            performance[task][i] = []
        for j, ilabels in enumerate(combinations(input_labels, i)):
            print(f"Training combination {j}")
            # train run 1
            if j >= len(performance[task][i]):
                performance[task][i].append(
                    cv_ridge_train_model(
                        df_always_train,
                        list_df_train_val,
                        df_test,
                        task,
                        ilabels,
                        alphas,
                        performance_evaluation=dict()))
            else:
                performance[task][i][j] = cv_ridge_train_model(
                    df_always_train, list_df_train_val, df_test, task,
                    ilabels, alphas, optimal_key="optimal",
                    performance_evaluation=performance[task][i][j])

            # find optimal value for alpha for second optimisation run
            for k, eval_metric in enumerate(["mean", "median"]):
                print(f"Optimising according to {eval_metric}")
                alphas2 = performance[task][i][j][f"optimal_{eval_metric}"]["alpha"]
                alphas2 = np.logspace(-1, 1, 10) * alphas2

                # train run 2
                performance[task][i][j] = cv_ridge_train_model(df_always_train, list_df_train_val, df_test, task,
                                                               ilabels, alphas2, optimal_key=f"optimal{k + 2}",
                                                               performance_evaluation=performance[task][i][j])

    return performance


def run_hpc_ridge_script(run, task, discard_interval):
    tasks_description = {
        "control": "all",
        "strawberry_1": "all",
        "strawberry_2": "all",
    }

    # src_dir = "/user/gent/420/vsc42053/reservoir_sensing/experiments/nature_plants_analysis"
    src_dir = "."

    print(f"Running {run}")

    for use_calibration in [True]:
        if use_calibration:
            print("Running with calibration")
        else:
            print("Running without calibration")

        sfn = os.path.join(src_dir, "cache")
        if not os.path.isdir(sfn):
            os.makedirs(os.path.join(sfn))
        if use_calibration:
            sfn = os.path.join(sfn, f"size_{run}_ridge_{24 - discard_interval * 2}h_{task.replace('/','-')}_results.pkl")
        else:
            sfn = os.path.join(sfn, f"size_nc_{run}_ridge_{24 - discard_interval * 2}h_{task.replace('/', '-')}_results.pkl")

        if os.path.isfile(sfn):
            print(f"{sfn} already exists. Skipping.")
            continue

        fn = os.path.join(src_dir, "data", f"{run}_data.csv")
        df = pd.read_csv(fn)

        print("Data loaded.")

        df_always_train, df_test, df_train_val = get_ridge_split(
            df,
            discard_interval=discard_interval)

        tasks = get_tasks(df_test, kind=tasks_description[run])

        print("Preprocessing done.")

        # get correct labels
        labels = deepcopy(input_labels[run])
        if not use_calibration:
            labels = [i.replace("_um", "_nc_au") for i in labels]

        print("Training...")

        print(f"Input labels: {labels}")
        print(f"Tasks: {tasks}")

        if os.path.isfile(sfn):
            performance = pickle.load(open(sfn, 'rb'))
            print("Loaded file from cache.")
        else:
            performance = dict()
        performance = train_ridge_task(
            df_always_train=df_always_train,
            list_df_train_val=df_train_val,
            df_test=df_test,
            task=task,
            input_labels=labels,
            alphas=np.power(10.0, np.arange(-10, 10)),
            size_analysis=None,
            performance=performance)

        print("\nStoring results.")

        pickle.dump(performance, open(sfn, 'wb'))


def merge_ridge_data():
    runs = ["control", "strawberry_1", "strawberry_2"]

    for use_calibration in [True, False]:
        for discard_interval in [2,4,6,9]:
            for i, r in enumerate(runs):
                print(f"Running {r}")

                tasks = [
                    "li6400xt_external_probe_air_temperature_C",
                    "li6400xt_external_probe_relative_humidity_percent",
                    "li6400xt_photosynthetic_rate_umol/m2/s",
                    "li6400xt_stomatal_conductance_mol/m2/s",
                    "li6400xt_transpiration_rate_mmol/m2/s",
                    "li6400xt_vapour_pressure_deficit_kPa",
                    #"li6400xt_sample_cell_air_temperature_C",
                    "li6400xt_leaf_temperature_C",
                    #"li6400xt_ref_cell_CO2_conc_umol/mol",
                    #"li6400xt_sample_cell_CO2_conc_umol/mol",
                    #"li6400xt_ref_cell_H2O_conc_mmol/mol",
                    #"li6400xt_sample_cell_H2O_conc_mmol/mol",
                    #"li6400xt_ref_cell_relative_humidity_conc_percent",
                    #"li6400xt_sample_cell_relative_humidity_conc_percent",
                    #"li6400xt_PAR_inside_chamber_umol/m2/s",
                    "li6400xt_PAR_outside_chamber_umol/m2/s",
                    #"li6400xt_air_pressure_kPa",
                    "light_sensor_lux",
                    #"soil_water_content_au",
                    "air_temperature_C",
                    "relative_humidity_percent"
                ]

                performance = dict()

                # load data per task
                for task in tasks:
                    if use_calibration:
                        sfn = os.path.join("cache", f"size_{r}_ridge_{24 - discard_interval * 2}h_{task.replace('/','-')}_results.pkl")
                    else:
                        sfn = os.path.join("cache", f"size_nc_{r}_ridge_{24 - discard_interval * 2}h_{task.replace('/','-')}_results.pkl")
                    if not os.path.isfile(sfn):
                        print(f"Unable to find {sfn}")
                        #continue
                    task_data = pickle.load(open(sfn, "rb"))
                    performance[task] = task_data[task]

                print("\nStoring results.")

                sfn = os.path.join("cache")
                if not os.path.isdir(sfn):
                    os.makedirs(os.path.join(sfn))
                if use_calibration:
                    sfn = os.path.join(sfn, f"size_{r}_ridge_{24-discard_interval*2}h_results.pkl")
                else:
                    sfn = os.path.join(sfn, f"size_nc_{r}_ridge_{24 - discard_interval * 2}h_results.pkl")

                print(f"Saving to {sfn}")
                pickle.dump(performance, open(sfn, 'wb'))

