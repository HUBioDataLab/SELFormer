def create_selfies_subset(selfies_df, save_to="./data/selfies_subset.txt", subset_size=100000):
	selfies_df.sample(frac=1).reset_index(drop=True)  # shuffling

	selfies_subset = selfies_df.selfies[:subset_size]
	selfies_subset = selfies_subset.to_frame()
	selfies_subset["selfies"].to_csv(save_to, index=False, header=False)

	print("Subset size {} saved to: {}".format(subset_size, save_to))
