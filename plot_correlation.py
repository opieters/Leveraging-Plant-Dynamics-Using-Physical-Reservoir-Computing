import matplotlib.pyplot as plt
import pandas as pd
from math import fabs
import numpy as np
from plot_ridge_analysis import short_name_map

if __name__ == "__main__":
    data_sources = [
        "control",
        "strawberry_1",
        "strawberry_2",
    ]

    for data_source in data_sources:
        df = pd.read_csv(f"data/{data_source}_mini_data.csv")

        plot_keys = [
            "li6400xt_PAR_outside_chamber_umol/m2/s",
            "air_temperature_C",
            "relative_humidity_percent",
            "li6400xt_photosynthetic_rate_umol/m2/s",
            # "li6400xt_stomatal_conductance_mol/m2/s",
            "li6400xt_transpiration_rate_mmol/m2/s",
            # "li6400xt_vapour_pressure_deficit_kPa",
            # "li6400xt_leaf_temperature_C",
            "leaf_thickness_1_um",
            "leaf_thickness_2_um",
            "leaf_thickness_3_um",
            "leaf_thickness_4_um",
            "leaf_thickness_5_um",
            "leaf_thickness_6_um",
            "leaf_thickness_7_um",
            "leaf_thickness_8_um",
        ]

        remove_keys = [i for i in df.keys() if i not in plot_keys]

        df = df.drop(columns=remove_keys)

        df = df.reindex(plot_keys, axis=1)

        corr = df.corr()

        xlabels = [short_name_map[i] if i in short_name_map  else i for i in corr.index]
        ylabels = [short_name_map[i] if i in short_name_map  else i for i in corr.keys()]

        plt.figure()
        plt.matshow(corr, vmin=-1, vmax=1)
        plt.title(data_source)
        plt.xticks(range(len(corr.index)), xlabels, rotation=90)
        plt.yticks(range(len(corr.columns)), ylabels)
        plt.colorbar()
        #plt.tight_layout()
        plt.show()

        with open(f"plot_data/correlation_{data_source}.dat", "w") as f:
            f.write("x y C\n")
            for j, y in enumerate(corr.keys()):
                for i, x in enumerate(corr.index):
                    if i <= j:
                        f.write(f"{i} {j} {fabs(corr.loc[x, y])}\n")
                    else:
                        f.write(f"{i} {j} nan\n")
                f.write("\n")

        average_corr = []
        n_corr = 0
        for j, y in enumerate(corr.keys()):
            for i, x in enumerate(corr.index):
                if ("leaf" in y) and (not ("leaf" in x)):
                    average_corr.append(fabs(corr.loc[x, y]))
                    n_corr += 1

        print(f"Average correlation: {np.mean(average_corr)} ({np.std(average_corr)} {n_corr})")


        corr.to_csv(f"plot_data/correlation_{data_source}.csv", index=True, index_label="index_label")
