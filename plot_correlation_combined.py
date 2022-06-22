import matplotlib.pyplot as plt
import pandas as pd
from math import fabs

from plot_ridge_analysis import short_name_map

if __name__ == "__main__":
    df1 = pd.read_csv(f"data/control_mini_data.csv")
    df2 = pd.read_csv(f"data/strawberry_1_mini_data.csv")

    plot_keys = [
        "li6400xt_PAR_outside_chamber_umol/m2/s",
        "air_temperature_C",
        "relative_humidity_percent",
        "li6400xt_photosynthetic_rate_umol/m2/s",
        #"li6400xt_stomatal_conductance_mol/m2/s",
        "li6400xt_transpiration_rate_mmol/m2/s",
        #"li6400xt_vapour_pressure_deficit_kPa",
        #"li6400xt_leaf_temperature_C",
        "leaf_thickness_1_um",
        "leaf_thickness_2_um",
        "leaf_thickness_3_um",
        "leaf_thickness_4_um",
        "leaf_thickness_5_um",
        "leaf_thickness_6_um",
        "leaf_thickness_7_um",
        "leaf_thickness_8_um",
    ]

    remove_keys = [i for i in df1.keys() if i not in plot_keys]

    df1 = df1.drop(columns=remove_keys)
    df2 = df2.drop(columns=remove_keys)
    df1 = df1.reindex(plot_keys, axis=1)
    df2 = df2.reindex(plot_keys, axis=1)

    corr1 = df1.corr()
    corr2 = df2.corr()

    with open(f"plot_data/correlation_combined.dat", "w") as f:
        f.write("x y C\n")
        for j, y in enumerate(corr1.keys()):
            for i, x in enumerate(corr1.index):
                if i < j:
                    f.write(f"{i} {j} {fabs(corr1.loc[x, y])}\n")
                else:
                    if i == j:
                        f.write(f"{i} {j} nan\n")
                    else:
                        f.write(f"{i} {j} {fabs(corr2.loc[x, y])}\n")
            f.write("\n")
