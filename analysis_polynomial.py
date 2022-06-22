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

from analysis_ridge import nmse, get_tasks, train_ridge_task, EvalPoint, cv_ridge_train_model, get_ridge_split, \
    input_labels


def run_hpc_nl_script(run, tasks, discard_interval, orders):

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
            sfn = os.path.join(sfn, f"nl_{run}_ridge_{24 - discard_interval * 2}h_results.pkl")
        else:
            sfn = os.path.join(sfn, f"nl_nc_{run}_ridge_{24 - discard_interval * 2}h_results.pkl")

        if os.path.isfile(sfn):
            print(f"{sfn} already exists. Skipping.")
            continue

        if os.path.isfile(sfn):
            delay_performance = pickle.load(open(sfn, 'rb'))
            print("Loaded file from cache.")
        else:
            delay_performance = dict()

        for order in orders:
            fn = os.path.join(src_dir, "data", f"{run}_data.csv")
            df = pd.read_csv(fn)

            print("Data loaded.")

            for task in tasks:
                x = df.loc[:, task].values
                x = (x - np.mean(x)) / np.std(x)
                x = np.power(x, order)
                df.loc[:, task] = x
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
                    size_analysis=7,
                    performance=performance)

            delay_performance[order] = performance

        print("\nStoring results.")

        pickle.dump(delay_performance, open(sfn, 'wb'))
