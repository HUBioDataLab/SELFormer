import pandas as pd
import chemprop

from pandarallel import pandarallel
import to_selfies


def smiles_to_selfies(df):
    df.insert(0, "selfies", df["smiles"])
    pandarallel.initialize()
    df.selfies = df.selfies.parallel_apply(to_selfies.to_selfies)

    df.drop(df[df.smiles == df.selfies].index, inplace=True)
    df.drop(columns=["smiles"], inplace=True)

    return df


def train_val_test_split_multilabel(path, scaffold_split):
    main_df = pd.read_csv(path)
    main_df.sample(frac=1).reset_index(drop=True)  # shuffling
    main_df.rename(columns={main_df.columns[0]: "smiles"}, inplace=True)
    main_df.fillna(0, inplace=True)
    main_df.reset_index(drop=True, inplace=True)

    if scaffold_split:
        molecule_list = []
        for _, row in main_df.iterrows():
            molecule_list.append(chemprop.data.data.MoleculeDatapoint(smiles=[row["smiles"]], targets=row[1:].values))
        molecule_dataset = chemprop.data.data.MoleculeDataset(molecule_list)
        (train, val, test) = chemprop.data.scaffold.scaffold_split(data=molecule_dataset, sizes=(0.8, 0.1, 0.1), seed=42, balanced=True)
        return (train, val, test)

    else:  # random split
        from sklearn.model_selection import train_test_split

        train, val = train_test_split(main_df, test_size=0.2, random_state=42)
        val, test = train_test_split(val, test_size=0.5, random_state=42)
        return (train, val, test)


def train_val_test_split(path, target_column_number=1, scaffold_split=False):
    main_df = pd.read_csv(path)
    main_df.sample(frac=1).reset_index(drop=True)  # shuffling
    main_df.rename(columns={main_df.columns[0]: "smiles", main_df.columns[target_column_number]: "target"}, inplace=True)
    main_df = main_df[["smiles", "target"]]
    # main_df.dropna(subset=["target"], inplace=True)
    main_df.fillna(0, inplace=True)
    main_df.reset_index(drop=True, inplace=True)

    if scaffold_split:
        molecule_list = []
        for _, row in main_df.iterrows():
            molecule_list.append(chemprop.data.data.MoleculeDatapoint(smiles=[row["smiles"]], targets=row[1:].values))
        molecule_dataset = chemprop.data.data.MoleculeDataset(molecule_list)
        (train, val, test) = chemprop.data.scaffold.scaffold_split(data=molecule_dataset, sizes=(0.8, 0.1, 0.1), seed=42, balanced=True)
        return (train, val, test)

    else:  # random split
        from sklearn.model_selection import train_test_split

        train, val = train_test_split(main_df, test_size=0.2, random_state=42)
        val, test = train_test_split(val, test_size=0.5, random_state=42)
        return (train, val, test)
