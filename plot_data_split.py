import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_sources = ["control", "strawberry_1", "strawberry_2"]
    zero_time = [
        pd.Timestamp("2021-09-16 16:21:45"),
        pd.Timestamp("2021-08-20 16:21:45"),
        pd.Timestamp("2021-09-30 16:21:45")
    ]
    end_time = [
        pd.Timestamp("2021-09-24 19:27:25"),
        pd.Timestamp("2021-08-28 19:27:25"),
        pd.Timestamp("2021-10-08 19:27:25")
    ]
    duration = (end_time[0] - zero_time[0]).total_seconds()

    print("Day boundaries:")
    t = pd.Timestamp("2021-09-17 00:00:00")
    while t < end_time[0]:
        pv = (t - zero_time[0]).total_seconds() / duration
        print(f"{pv},")
        t += pd.Timedelta("1 days")

    for i_ds, data_source in enumerate(data_sources):
        print(f"Split for {data_source}")
        df = pd.read_csv(f"data/{data_source}_data.csv")

        df.loc[:,"time"] = pd.to_datetime(df["time"])

        time_offset = (df.loc[0, "time"] - zero_time[i_ds]).total_seconds()

        #discard_mask = df["train_val_test_split"].values == 4
        #discard_mask = np.logical_and(discard_mask, np.logical_not(df["di_4"].values))

        #plt.plot(df.loc[:,"time"], df.loc[:, "air_temperature_C"])
        #plt.plot(df.loc[discard_mask, "time"], df.loc[discard_mask, "air_temperature_C"])
        #plt.show()
        #exit()

        # detect train mask
        mask = df["train_val_test_split"].values
        mask = mask >= 0

        mask = np.logical_and(mask, np.logical_not(df["di_4"].values))
        train_split = []

        if mask[0] == True:
            train_split.append([time_offset])

        for i in range(len(mask) - 1):
            mask_cut = tuple(mask[i:i + 2])
            if mask_cut == (True, False):
                # transition from test to train/val mask
                train_split[-1].append(i + time_offset)
            if mask_cut == (False, True):
                train_split.append([i + 1 + time_offset])
        if mask[-1]:
            train_split[-1].append(len(df) - 1 + time_offset)

        # detect test mask
        mask = df["train_val_test_split"].values
        mask = mask < 0

        mask = np.logical_and(mask, np.logical_not(df["di_4"].values))

        test_split = []

        if mask[0] == True:
            test_split.append([time_offset])

        for i in range(len(mask)-1):
            mask_cut = tuple(mask[i:i+2])
            if mask_cut == (True, False):
                # transition from test to train/val mask
                test_split[-1].append(i + time_offset)
            if mask_cut == (False, True):
                test_split.append([i + 1 + time_offset])
        if mask[-1]:
            test_split[-1].append(len(df)-1 + time_offset)

        # print test split
        print("Train:")
        for i in train_split:
            print(f"{i[0]/duration}/{i[1]/duration},")
        print("Test:")
        for i in test_split:
            print(f"{i[0]/duration}/{i[1]/duration},")