import pandas as pd


def create_selfies_subset(selfies_df, subset_size=100000):
    selfies_df.sample(frac=1).reset_index(drop=True) # shuffling

    selfies_subset = selfies_df.selfies[:subset_size]
    selfies_subset = selfies_subset.to_frame()
    selfies_subset["selfies"].to_csv("./data/selfies_subset.txt", index=False, header=False)

    print("Subset size {} saved to: {}".format(subset_size, "./data/selfies_subset.txt"))