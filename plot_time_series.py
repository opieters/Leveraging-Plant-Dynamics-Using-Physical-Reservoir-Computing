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
from analysis_narma import narma, nmse

if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (20, 15)

    data_sources = [
        "control",
        "strawberry_1",
        "strawberry_2",
    ]

    tasks = [
        'li6400xt_photosynthetic_rate_umol/m2/s',
        #'li6400xt_stomatal_conductance_mol/m2/s',
        'li6400xt_transpiration_rate_mmol/m2/s',
        #'li6400xt_vapour_pressure_deficit_kPa',
        #'li6400xt_leaf_temperature_C',
        'li6400xt_PAR_outside_chamber_umol/m2/s',
        #'air_temperature_C',
        #'relative_humidity_percent'
        #"NARMA20-li6400xt_PAR_outside_chamber_umol/m2/s"
        #"delay-200-li6400xt_PAR_outside_chamber_umol/m2/s",
        #"delay-10000-li6400xt_PAR_outside_chamber_umol/m2/s",
        #"delay-5000-li6400xt_PAR_outside_chamber_umol/m2/s",
    ]


    input_labels = [
        "leaf_thickness_1_um", "leaf_thickness_2_um", "leaf_thickness_3_um", "leaf_thickness_4_um",
        "leaf_thickness_5_um", "leaf_thickness_6_um", "leaf_thickness_7_um", "leaf_thickness_8_um"
    ]

    discard_interval = 4

    fig_test, axs_test = plt.subplots(len(tasks), len(data_sources), squeeze=False)
    # fig_train, axs_train = plt.subplots(len(discard_intervals), len(data_sources), sharex=True, sharey=True)
    for idx_x, data_source in enumerate(data_sources):
        for idx_y, task in enumerate(tasks):
            plot_df = dict()

            original_task_to_plot = task
            if "NARMA" in task:
                narma_task = task.split("-")
                narma_order = int(narma_task[0].replace("NARMA", ""))
                narma_source = narma_task[1]
                narma_task = f"NARMA{narma_order}"

                sfn = os.path.join("./cache", f"narma_{data_source}_ridge_{24 - discard_interval * 2}h_{narma_source.replace('/', '-')}_results.pkl")
            elif "delay" in task:
                delay_task = task.split("-")
                delay_value = int(delay_task[1])
                delay_source = delay_task[2]
                sfn = os.path.join("./cache", f"delay_{data_source}_ridge_{24 - discard_interval*2}h_results.pkl")
            else:
                sfn = os.path.join("./cache", f"size_{data_source}_ridge_{24 - discard_interval * 2}h_results.pkl")

            if not os.path.isfile(sfn):
                print(f"Model {sfn} not found.")
                continue

            # load models
            performance = pickle.load(open(sfn, 'rb'))

            # load data
            if "NARMA" in task:
                fn = os.path.join(f"./data/{data_source}_mini_data.csv")
            elif "delay" in task:
                fn = os.path.join(f"./data/{data_source}_data.csv")
            else:
                #fn = os.path.join(f"./data/{data_source}_mini_data.csv")
                fn = os.path.join(f"./data/{data_source}_mini_data.csv")
            df = pd.read_csv(fn)

            # generate NARMA task if required
            if "NARMA" in task:
                task = narma_task
                df = df.assign(**{task: narma(df[narma_source].values, k=narma_order)})
            if "delay" in task:
                df_shift = df[[delay_source]].shift(periods=delay_value, axis="index")
                for k in df_shift:
                    df = df.assign(**{k: df_shift[k]})
                print(f"Going for {data_source} from {len(df)} ", end="")
                df.dropna(inplace=True)
                print(f"to {len(df)}")

            if "delay" in task:
                task = delay_source
                df = df.drop(index=np.arange(delay_value, 200))

                performance = performance[delay_value]
            else:
                df = df.drop(index=np.arange(0, 200))

            df.loc[:, "time"] = pd.to_datetime(df["time"])
            df.set_index("time", drop=True, inplace=True)

            # write plot data to file
            df_traces = dict()

            x = df.index

            y = df[task]

            size_keys = performance[task].keys()
            size_keys = sorted(list(size_keys))
            if len(size_keys) > 1:
                size_key = size_keys[-1]
            else:
                size_key = size_keys[0]
            print(f"Using {size_keys} clips")

            y_hats = []

            # compute prediction with mean and standard error
            size_evals = performance[task][size_key]
            for j in range(len(size_evals)):
                if len(size_evals) > 1:
                    x_model = size_evals[j]["labels"]
                    #x_model = [i.replace("_um", "_nc_au") for i in x_model]
                    x_model = [df[i].values for i in x_model]
                else:
                    x_model = [i for i in input_labels]
                    #x_model = [i.replace("_um", "_nc_au") for i in x_model]
                    x_model = [df[i] for i in x_model]
                x_model = np.array(x_model)
                x_model = np.transpose(x_model)
                # compute prediction if all but one input traces are used
                y_hats.append(size_evals[j]["optimal_median"]["model"].predict(x_model))
                coefficients = size_evals[j]["optimal_median"]["model"]["ridge"].coef_
                print(f'Coefficients: {coefficients}')
                print(f"Stats: {np.mean(coefficients)} {np.std(coefficients)}")
            if len(y_hats) > 1:
                y_mean = np.mean(y_hats, axis=0)
                y_err = np.std(y_hats, axis=0)
            else:
                y_mean = y_hats[0]
                y_err = 0

                nmse_value = size_evals[j]["optimal_median"]["nmse"].test
                nmse_plot = nmse(yh=y_mean, y=y)
                print(f"{data_source} - {task} NMSE: {nmse_value}, plot: {nmse_plot}")
                print(f"Number of test samples {np.sum(df['train_val_test_split'].values == -1)}/{len(df)}")


            # compute prediction if all input traces are used
            #x_model = [df[i] for i in input_labels]
            #x_model = np.array(x_model)
            #x_model = np.transpose(x_model)
            #size_evals = performance[task][size_key]
            #y_max_inputs = size_evals[0]["optimal_median"]["model"].predict(x_model)

            axs_test[idx_y, idx_x].plot(x, y, "g", label="y_true")
            axs_test[idx_y, idx_x].plot(x, y_mean, "r", label="y_hat")
            #axs_test[idx_y, idx_x].plot(x, y_max_inputs, "k", label="y_max")
            #axs_test[idx_y, idx_x].fill_between(x, y_mean - y_err, y_mean + y_err, alpha=0.2)

            axs_test[idx_y, idx_x].set_title(f"TE {data_source} {24 - discard_interval * 2}h")
            axs_test[idx_y, idx_x].set_xlabel("time")
            if "NARMA" not in task:
                axs_test[idx_y, idx_x].set_ylabel(short_name_map[task])
            axs_test[idx_y, idx_x].grid()

            plot_df["time"] = x
            plot_df[f"y_true_{task}"] = y
            plot_df[f"y_pred_{task}"] = y_mean

            plot_df = pd.DataFrame(plot_df)
            plot_df.to_csv(f"plot_data/time_trace_{original_task_to_plot.replace('/', '-')}_{data_source}.csv", index=False)

    plt.show()
    fig_test.savefig(f"plots/delay_plot_{original_task_to_plot.replace('/', '-')}_{data_source}.png", dpi=300)