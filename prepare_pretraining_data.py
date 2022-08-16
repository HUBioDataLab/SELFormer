import pandas as pd
from pandarallel import pandarallel

import to_selfies


def prepare_data(path="data/molecule_dataset_smiles.txt", save_to="./data/molecule_dataset_selfies.csv"):
    chembl_df = pd.read_csv(path, sep="\t")  # data is TAB separated.
    # chembl_df.drop(columns=["standard_inchi", "standard_inchi_key"], inplace=True)  # we are not interested in "standard_inchi" and "standard_inchi_key" columns.
    chembl_df["selfies"] = chembl_df["canonical_smiles"]  # creating a new column "selfies" that is a copy of "canonical_smiles"

    pandarallel.initialize()
    chembl_df.selfies = chembl_df.selfies.parallel_apply(to_selfies.to_selfies)

    chembl_df.drop(chembl_df[chembl_df.canonical_smiles == chembl_df.selfies].index, inplace=True)
    chembl_df.drop(columns=["canonical_smiles"], inplace=True)
    chembl_df.to_csv(save_to, index=False)


def create_selfies_file(selfies_df, save_to="./data/selfies_subset.txt", subset_size=100000, do_subset=True):
    selfies_df.sample(frac=1).reset_index(drop=True)  # shuffling

    if do_subset:
        selfies_subset = selfies_df.selfies[:subset_size]
    else:
        selfies_subset = selfies_df.selfies
    selfies_subset = selfies_subset.to_frame()
    selfies_subset["selfies"].to_csv(save_to, index=False, header=False)
