from analysis_delay import run_hpc_delay_script
from analysis_narma import run_hpc_narma_script
from analysis_polynomial import run_hpc_nl_script
import numpy as np

from analysis_ridge import run_hpc_ridge_script, merge_ridge_data


def run_delay_analysis(discard_intervals, runs):
    tasks = [
        "li6400xt_PAR_outside_chamber_umol/m2/s",
    ]

    for run in runs:
        for discard_interval in discard_intervals:
            if discard_interval > 2:
                delays = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            else:
                delays = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

            run_hpc_delay_script(run=run, tasks=tasks, discard_interval=discard_interval, delays=delays)


def run_narma_analysis(discard_intervals, runs):
    tasks = [
        "li6400xt_PAR_outside_chamber_umol/m2/s",
    ]

    for run in runs:
        for discard_interval in discard_intervals:
            for task in tasks:
                run_hpc_narma_script(run=run, task=task, discard_interval=discard_interval,
                                     narma_orders=[2, 5, 10, 20, 50, 100])


def run_analysis_polynomial(discard_intervals, runs):

    orders = np.arange(1, 10, dtype=int)

    tasks = [
        "li6400xt_PAR_outside_chamber_umol/m2/s",
        "light_sensor_lux",
    ]

    for run in runs:
        for discard_interval in discard_intervals:
            run_hpc_nl_script(run=run, tasks=tasks, discard_interval=discard_interval, orders=orders)


def run_ridge_analysis(discard_intervals, runs):
    tasks = [
        "li6400xt_external_probe_air_temperature_C",
        "li6400xt_external_probe_relative_humidity_percent",
        "li6400xt_photosynthetic_rate_umol/m2/s",
        "li6400xt_transpiration_rate_mmol/m2/s",
        "li6400xt_leaf_temperature_C",
        "li6400xt_PAR_outside_chamber_umol/m2/s",
        "light_sensor_lux",
        "air_temperature_C",
        "relative_humidity_percent"
    ]
    for run in runs:
        for discard_interval in discard_intervals:
            for task in tasks:
                run_hpc_ridge_script(run=run, task=task, discard_interval=discard_interval)

    merge_ridge_data()

if __name__ == "__main__":
    run_delay_analysis(runs=["control"], discard_intervals=[4])
    run_narma_analysis(runs=["strawberry_1", "strawberry_2", "control"], discard_intervals=[4])
    run_analysis_polynomial(runs=["strawberry_1", "strawberry_2", "control"], discard_intervals=[4])
    run_ridge_analysis(runs=["strawberry_1", "strawberry_2", "control"], discard_intervals=[4])