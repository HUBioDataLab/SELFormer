import selfies as sf


def get_selfies_alphabet(chembl_df, path="./data/chembl_29_selfies_alphabet.txt"):
	selfies_array = chembl_df.selfies.to_numpy(copy=True)
	selfies_alphabet = sf.get_alphabet_from_selfies(selfies_array)

	with open(path, "w") as f:
		f.write(",".join(list(selfies_alphabet)))
