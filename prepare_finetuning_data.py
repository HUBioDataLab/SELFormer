import pandas as pd
import chemprop

from pandarallel import pandarallel
import to_selfies

# def fix_esol(path):
#     main_df = pd.read_csv(path)
#     main_df.insert(0, "smiles", main_df.pop("smiles"))
#     main_df.to_csv(path, index=False)
# fix_esol("data/finetuning/regression/esol/esol.csv")

def smiles_to_selfies(df, out_file):
	df["selfies"] = df["smiles"]
	pandarallel.initialize()
	df.selfies = df.selfies.parallel_apply(to_selfies.to_selfies)

	df.drop(df[df.smiles == df.selfies].index, inplace=True)
	df.drop(columns=["smiles"], inplace=True)
	df = df[["selfies", "target"]]
	df.to_csv(out_file, index=False)

	print(".csv file saved to: " + out_file)

    # return df


def train_val_test_split(path):
    main_df = pd.read_csv(path)
    main_df.sample(frac=1).reset_index(drop=True)  # shuffling
    main_df.rename(columns={main_df.columns[0]: "smiles", main_df.columns[1]: "target"}, inplace=True)

    molecule_list = []
    for _, row in main_df.iterrows():
        molecule_list.append(chemprop.data.data.MoleculeDatapoint(smiles=[row["smiles"]], targets=row["target"]))
    molecule_dataset = chemprop.data.data.MoleculeDataset(molecule_list)

    (train, val, test) = chemprop.data.scaffold.scaffold_split(data=molecule_dataset, sizes=(0.8, 0.1, 0.1), seed=42, balanced=True)

    return (train, val, test)

#     datasets = {"train": train, "val": val, "test": test}

#     for key in datasets.keys():
#         current_df = pd.DataFrame(datasets[key].smiles(flatten=True), columns=["smiles"])
#         temp_df = pd.DataFrame()
#         for i in datasets[key].targets():
#             temp_df = temp_df.append(i.to_frame().T, ignore_index=True)
#         current_df = current_df.join(temp_df)

#         current_df["selfies"] = current_df["smiles"]  # creating a new column "selfies" that is a copy of "canonical_smiles"

#         pandarallel.initialize()
#         current_df.selfies = current_df.selfies.parallel_apply(to_selfies.to_selfies)

#         current_df.drop(current_df[current_df.smiles == current_df.selfies].index, inplace=True)
#         current_df.drop(columns=["smiles"], inplace=True)
#         current_df.insert(0, "selfies", current_df.pop("selfies"))

#         current_df.to_csv(path + "_" + key + ".csv", index=False)

# classification_dataset_paths = ["data/finetuning/classification/bace/bace.csv", "data/finetuning/classification/bbbp/bbbp.csv", "data/finetuning/classification/hiv/hiv.csv", "data/finetuning/classification/sider/sider.csv", "data/finetuning/classification/tox21/tox21.csv"]
# regression_dataset_paths = ["data/finetuning/regression/esol/esol.csv", "data/finetuning/regression/freesolv/freesolv.csv", "data/finetuning/regression/lipo/lipo.csv", "data/finetuning/regression/pdbbind_full/pdbbind_full.csv"]

# for path in classification_dataset_paths:
#     train_val_test_split(path)

# for path in regression_dataset_paths:
#     train_val_test_split(path)