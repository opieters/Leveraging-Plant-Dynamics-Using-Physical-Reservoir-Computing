import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plot_ridge_analysis import short_name_map

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation.
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    datay = datay.shift(lag)

    return datax.corr(datay)

if __name__ == "__main__":
    data_sources = [
        #"control",
        "strawberry_1",
        #"strawberry_2",
    ]

    xcorr_combinations = [
        #("air_temperature_C", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        #("leaf_thickness_1_um", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        #("leaf_thickness_2_um", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        #("leaf_thickness_3_um", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        #("leaf_thickness_4_um", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        #("leaf_thickness_5_um", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        #("leaf_thickness_6_um", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        #("leaf_thickness_7_um", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        #("leaf_thickness_8_um", "li6400xt_PAR_outside_chamber_umol/m2/s"),
        ("leaf_thickness_8_um", "leaf_thickness_1_um"),
        ("leaf_thickness_1_um", "leaf_thickness_2_um"),
        ("leaf_thickness_2_um", "leaf_thickness_3_um"),
        ("leaf_thickness_3_um", "leaf_thickness_4_um"),
        ("leaf_thickness_4_um", "leaf_thickness_5_um"),
        ("leaf_thickness_5_um", "leaf_thickness_6_um"),
        ("leaf_thickness_6_um", "leaf_thickness_5_um"),
        ("leaf_thickness_7_um", "leaf_thickness_8_um"),
    ]

    xcorr_combinations = []
    for i in range(8):
        for j in range(8):
            if i == j:
                break
            else:
                xcorr_combinations.append(
                    (f"leaf_thickness_{i+1}_um", f"leaf_thickness_{j+1}_um")
                )

    for data_source in data_sources:
        print(f"Loading data source {data_source}")
        df = pd.read_csv(f"data/{data_source}_data.csv")

        cutoff = int(24*60*60)

        fig = plt.figure()

        dfn = dict()

        for i, xcorr_combination in enumerate(xcorr_combinations):
            x = df[xcorr_combination[0]]
            y = df[xcorr_combination[1]]

            lags = np.linspace(0, cutoff, 200, dtype=int)
            time = lags / 60 / 60
            c = [crosscorr(x, y, lag) for lag in lags]
            c = np.abs(c)

            if "time" not in dfn:
                dfn["time"] = time
            dfn[f"{i}"] = c

            plt.plot(time, c, label=f"{i}")

            print(f"Max for {xcorr_combination} at {lags[np.argmax(c)]} of {np.max(c)}")
        #plt.legend()
        fig.savefig(f"plots/plt_xcorr_{data_source}.png", facecolor='white', edgecolor='none')
        plt.show()

        dfn = pd.DataFrame(dfn)

        dfn.to_csv(f"plot_data/xcorr_{data_source}.csv", index=False)