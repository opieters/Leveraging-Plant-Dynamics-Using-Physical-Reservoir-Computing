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

from analysis_ridge import nmse, get_tasks, train_ridge_task, EvalPoint, cv_ridge_train_model, get_ridge_split, input_labels


def narma(x, k, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1):
    y = np.zeros(len(x))

    x = 0.2*(x - np.mean(x)) / np.max(np.abs(x))

    for i in range(k, len(x)):
        y[i] = alpha * y[i-1] + beta / k * y[i-1]*np.sum(y[i-k:i]) + gamma * x[i-k] * x[i] + delta
    return y


def narma2(x, k, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1):
    y = np.zeros(len(x))

    x = 0.2*(x - np.mean(x)) / np.max(np.abs(x))

    for i in range(60*k, len(x)):
        y[i] = alpha * y[i-1] + beta / k * y[i-1]*np.sum(y[i-60*k:i:60]) + gamma * x[i-60*k] * x[i-1] + delta
    return y


def run_hpc_narma_script(run, task, discard_interval, narma_orders):
    tasks_description = {
        "control": "all",
        "strawberry_1": "all",
        "strawberry_2": "all",
    }

    # src_dir = "/user/gent/420/vsc42053/reservoir_sensing/experiments/nature_plants_analysis"
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
            sfn = os.path.join(sfn, f"narma_{run}_ridge_{24 - discard_interval * 2}h_{task.replace('/','-')}_results.pkl")
        else:
            sfn = os.path.join(sfn, f"narma_nc_{run}_ridge_{24 - discard_interval * 2}h_{task.replace('/', '-')}_results.pkl")

        if os.path.isfile(sfn):
            print(f"{sfn} already exists. Skipping.")
            continue

        fn = os.path.join(src_dir, "data", f"{run}_data.csv")
        df = pd.read_csv(fn)

        print("Data loaded.")

        narma_tasks = []
        for narma_order in narma_orders:
            narma_task = f"NARMA{narma_order}"
            df[narma_task] = narma2(df[task].values, k=narma_order)
            #df[narma_task] = narma(df[task].values, k=narma_order)
            narma_tasks.append(narma_task)

        print("NARMA data generated.")

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

        if os.path.isfile(sfn):
            performance = pickle.load(open(sfn, 'rb'))
            print("Loaded file from cache.")
        else:
            performance = dict()
        for narma_task in narma_tasks:
            performance = train_ridge_task(
                df_always_train=df_always_train,
                list_df_train_val=df_train_val,
                df_test=df_test,
                task=narma_task,
                input_labels=labels,
                alphas=np.power(10.0, np.arange(-10, 10)),
                size_analysis=[7, 8],
                performance=performance)

        print("\nStoring results.")

        pickle.dump(performance, open(sfn, 'wb'))
