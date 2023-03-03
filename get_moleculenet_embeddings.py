import os
from time import time
from fnmatch import fnmatch

import pandas as pd
from pandarallel import pandarallel
import to_selfies
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, metavar="/path/to/dataset/", help="Path of the input MoleculeNet datasets.")
parser.add_argument("--model_file", required=True, metavar="<str>", type=str, help="Name of the pretrained model.")

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_file = args.model_file # path of the pre-trained model
config = RobertaConfig.from_pretrained(model_file)
config.output_hidden_states = True
tokenizer = RobertaTokenizer.from_pretrained("./data/RobertaFastTokenizer")
model = RobertaModel.from_pretrained(model_file, config=config)


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


def generate_embeddings(model_file, args):

    root = args.dataset_path
    model_name = model_file.split("/")[-1]

    prepare_data_pattern = "*.csv"

    print(f"\nGenerating embeddings using pre-trained model {model_name}")
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, prepare_data_pattern) and not any(substring in name for substring in ['selfies', 'embeddings', 'results']):
                dataset_file = os.path.join(path, name)
                generate_moleculenet_selfies(dataset_file)

                selfies_file = os.path.join(path, name.split(".")[0] + "_selfies.csv")

                dataset_name = selfies_file.split("/")[-1].split(".")[0].split('_selfies')[0]
                print(f'\nGenerating embeddings for {dataset_name}')
                t0 = time()

                dataset_df = pd.read_csv(selfies_file)
                pandarallel.initialize(nb_workers=10, progress_bar=True) # number of threads
                dataset_df["sequence_embeddings"] = dataset_df.selfies.parallel_apply(get_sequence_embeddings, args=(tokenizer, model))

                dataset_df.drop(columns=["selfies"], inplace=True) # not interested in selfies data anymore, only class and the embedding
                file_name = f"{dataset_name}_{model_name}_embeddings.csv"

                # save embeddings to file
                path = os.path.dirname(selfies_file)
                dataset_df.to_csv(os.path.join(path, file_name), index=False)
                t1 = time()

                print(f'Finished in {round((t1-t0) / 60, 2)} mins')
                print(f'Saved to {os.path.join(path, file_name)}\n')

generate_embeddings(model_file, args)