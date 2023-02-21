import os
from time import time
import pandas as pd
import numpy as np
from pandarallel import pandarallel
import to_selfies
import torch

def generate_moleculenet_selfies(dataset_file):
    """
    Generates SELFIES for a given dataset and saves it to a file.
    :param dataset_file: path to the dataset file
    """

    dataset_name = dataset_file.split("/")[-1].split(".")[0]
    
    print(f'\nGenerating SELFIES for {dataset_name}')

    if dataset_name == 'bace':
        smiles_column = 'mol'
    else:
        smiles_column = 'smiles'

    # read dataset
    dataset_df = pd.read_csv(os.path.join(dataset_file))
    dataset_df["selfies"] = dataset_df[smiles_column] # creating a new column "selfies" that is a copy of smiles_column

    # generate selfies
    pandarallel.initialize()
    dataset_df.selfies = dataset_df.selfies.parallel_apply(to_selfies.to_selfies)

    dataset_df.drop(dataset_df[dataset_df[smiles_column] == dataset_df.selfies].index, inplace=True)
    dataset_df.drop(columns=[smiles_column], inplace=True)
    out_name = dataset_name + "_selfies.csv"

    # save selfies to file
    path = os.path.dirname(dataset_file)

    dataset_df.to_csv(os.path.join(path, out_name), index=False)
    print(f'Saved to {os.path.join(path, out_name)}')


def get_sequence_embeddings(selfies, tokenizer, model):

    torch.set_num_threads(1)
    token = torch.tensor([tokenizer.encode(selfies, add_special_tokens=True, max_length=512, padding=True, truncation=True)])
    output = model(token)

    sequence_out = output[0]
    return torch.mean(sequence_out[0], dim=0).tolist()


def generate_moleculenet_embeddings(selfies_file, which_model, tokenizer, model):

    """ 
    Generates embeddings for a given dataset and saves it to a file.
    :param selfies_file: path to the dataset file
    :param which_model: model name 
    :param tokenizer: tokenizer for the model
    :param model: model to use for generating embeddings
    """
    
    dataset_name = selfies_file.split("/")[-1].split(".")[0].split('_selfies')[0]
    print(f'\nGenerating embeddings for {dataset_name}')
    t0 = time()

    dataset_df = pd.read_csv(selfies_file)
    pandarallel.initialize(nb_workers=10, progress_bar=True) # number of threads
    dataset_df["sequence_embeddings"] = dataset_df.selfies.parallel_apply(get_sequence_embeddings, args=(tokenizer, model))

    dataset_df.drop(columns=["selfies"], inplace=True) # not interested in selfies data anymore, only class and the embedding
    file_name = f"{dataset_name}_{which_model}_embeddings.csv"

    # save embeddings to file
    path = os.path.dirname(selfies_file)
    dataset_df.to_csv(os.path.join(path, file_name), index=False)
    t1 = time()

    print(f'Finished in {round((t1-t0) / 60, 2)} mins')
    print(f'Saved to {os.path.join(path, file_name)}\n')


def train_val_test_split_multilabel(path): # random split
    """
    Randomly splits a multi-label classification dataset into train and test sets.
    :param path: path to the dataset file
    :return: train, test sets
    """

    main_df = pd.read_csv(path)
    main_df['sequence_embeddings'] = main_df['sequence_embeddings'].str.strip('[]').str.split(',').apply(lambda x: np.array([float(i) for i in x]))
    main_df.sample(frac=1).reset_index(drop=True)  # shuffling
    main_df.fillna(0, inplace=True)
    main_df.reset_index(drop=True, inplace=True)

    from sklearn.model_selection import train_test_split

    train, val = train_test_split(main_df, test_size=0.2, random_state=42)
    val, test = train_test_split(val, test_size=0.5, random_state=42)
    # combine train and valid set as SVMs don't use a validation set, but NNs do.
    # this way they use the same amount of data.
    train = pd.concat([train, val])

    return (train, test)


def train_val_test_split(path, target_column_name): # random split
    """
    Randomly splits a classification or regression dataset into train and test sets.
    :param path: path to the dataset file
    :param target_column_name: name of the column that contains the target values
    :return: train, test sets
    """

    main_df = pd.read_csv(path)
    main_df['sequence_embeddings'] = main_df['sequence_embeddings'].str.strip('[]').str.split(',').apply(lambda x: np.array([float(i) for i in x]))
    main_df.sample(frac=1).reset_index(drop=True)  # shuffling
    main_df.rename(columns={target_column_name: "target"}, inplace=True)
    main_df = main_df[["sequence_embeddings", "target"]]
    # main_df.dropna(subset=["target"], inplace=True)
    main_df.fillna(0, inplace=True)
    main_df.reset_index(drop=True, inplace=True)

    from sklearn.model_selection import train_test_split

    train, val = train_test_split(main_df, test_size=0.2, random_state=42)
    val, test = train_test_split(val, test_size=0.5, random_state=42)

    # combine train and valid set as SVMs don't use a validation set, but NNs do.
    # this way they use the same amount of data.
    train = pd.concat([train, val])

    return (train, test)