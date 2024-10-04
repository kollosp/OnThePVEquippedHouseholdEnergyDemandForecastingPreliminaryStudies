import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns



SECS_PER_DAY = 60*60*24
NB_BINS = 90
Y_BINS = 90

BIN_LEN = SECS_PER_DAY / NB_BINS


def load_dataset(which):
    #
    # "datasets/pv.csv"

    if which == 1:
        df = pd.read_excel("datasets/MOK.xlsx")
        df = df[["X0", "X2"]]  # filter timestamps and energy demand

    else:
        df = pd.read_csv("datasets/pv.csv", sep=",", names=["X0", "X2"], header=None)

    df["day_counter"] = df['X0'] // SECS_PER_DAY
    df["day_counter"] = df["day_counter"] - df["day_counter"].min()
    df["chunk"] = (df['X0'].mod(SECS_PER_DAY) // BIN_LEN).astype(int)
    df["X0"] = pd.to_datetime(df['X0'], unit='s')
    df.set_index("X0", inplace=True)

    return df

def prep_split():

    return df

def chunk_separation(df):
    chunks = df["chunk"].unique()
    dfs = []
    for chunk in chunks:
        current = df.loc[df["chunk"] == chunk]
        current = current.groupby(by="day_counter").mean()

        dfs.append(current)
    return dfs

def day_counter_separation(df):
    chunks = df["day_counter"].unique()
    dfs = []
    for chunk in chunks:
        current = df.loc[df["day_counter"] == chunk]
        current = current.groupby(by="chunk").mean()

        dfs.append(current)
    return dfs

if __name__ == "__main__":
    data = load_dataset(which=1)
    mx = data["X2"].max()
    dfs = day_counter_separation(data)

    for df in dfs:
        plt.plot(df.index, df["X2"]/mx, alpha=.1, c="b")
    plt.xlabel("Time chunk")
    plt.ylabel("Normalized Power")

    # dfs = chunk_separation(data)
    # heatmap = np.zeros((Y_BINS, NB_BINS))
    # for i, df in enumerate(dfs):
    #     s = df["X2"].to_numpy()
    #     heatmap[:, i], _ = np.histogram(s, Y_BINS)
    #     # heatmap[:, i] = heatmap[:, i] / heatmap[:, i].max()
    #
    # ax = sns.heatmap(heatmap)
    # ax.invert_yaxis()

    # data = f()
    #
    #
    #
    # for i, df in enumerate(dfs):
    #     print(i, len(df))


    # df = df[0:288*10]

    plt.show()

