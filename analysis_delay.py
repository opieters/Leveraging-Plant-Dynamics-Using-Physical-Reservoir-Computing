#!/usr/bin/env python3
from copy import deepcopy

import pandas as pd
import pickle
import numpy as np
import os
from analysis_ridge import train_ridge_task, get_ridge_split, input_labels

def narma(x, k, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1):
    y = np.zeros(len(x))

    x = (x - np.mean(x)) / np.max(np.abs(x))

    for i in range(k, len(x)):
        y[i] = alpha * y[i-1] + beta / k * y[i-1]*np.sum(y[i-k:i]) + gamma * x[i-k] * x[i] + delta
    return y


def run_hpc_delay_script(run, tasks, discard_interval, delays):
    src_dir = "."

    print(f"Running {run}")

    for use_calibration in [True, False]:
        if use_calibration:
            print("Running with calibration")
        else:
            print("Running without calibration")

        sfn = os.path.join(src_dir, "cache")
        if not os.path.isdir(sfn):
            os.makedirs(os.path.join(sfn))
        if use_calibration:
            sfn = os.path.join(sfn, f"delay_{run}_ridge_{24 - discard_interval * 2}h_results.pkl")
        else:
            sfn = os.path.join(sfn, f"delay_nc_{run}_ridge_{24 - discard_interval * 2}h_results.pkl")

        if os.path.isfile(sfn):
            print(f"{sfn} already exists. Skipping.")
            continue

        if os.path.isfile(sfn):
            delay_performance = pickle.load(open(sfn, 'rb'))
            print("Loaded file from cache.")
        else:
            delay_performance = dict()

        for delay in delays:
            fn = os.path.join(src_dir, "data", f"{run}_data.csv")
            df = pd.read_csv(fn)

            print("Data loaded.")

            not_tasks = [i for i in df.keys() if i not in tasks]
            df_shift = df[tasks].shift(periods=delay, axis="index")
            df = df[not_tasks]
            df = df.join(other=df_shift)
            df.dropna(inplace=True)

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
                performance = train_ridge_task(
                    df_always_train=df_always_train,
                    list_df_train_val=df_train_val,
                    df_test=df_test,
                    task=task,
                    input_labels=labels,
                    alphas=np.power(10.0, np.arange(-10, 10)),
                    size_analysis=[7, 8],
                    performance=performance)

            delay_performance[delay] = performance

        print("\nStoring results.")

        pickle.dump(delay_performance, open(sfn, 'wb'))
