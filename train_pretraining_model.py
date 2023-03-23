import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--smiles_dataset", required=False, metavar="/path/to/dataset/", help="Path of the SMILES dataset. If you provided --selfies_dataset argument, then this argument is not required.")
parser.add_argument("--selfies_dataset", required=True, metavar="/path/to/dataset/", help="Path of the SEFLIES dataset. If it does not exist, it will be created at the given path.")
parser.add_argument("--subset_size", required=False, metavar="<int>", type=int, default=0, help="By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset.")
parser.add_argument("--prepared_data_path", required=True, metavar="/path/to/dataset/", help="Path of the .txt prepared data. If it does not exist, it will be created at the given path.")
parser.add_argument("--bpe_path", required=True, metavar="/path/to/bpetokenizer/", default="", help="Path of the BPE tokenizer. If it does not exist, it will be created at the given path.")
parser.add_argument("--roberta_fast_tokenizer_path", required=True, metavar="/path/to/robertafasttokenizer/", help="Directory of the RobertaTokenizerFast tokenizer. RobertaFastTokenizer only depends on the BPE Tokenizer and will be created regardless of whether it exists or not.")
parser.add_argument("--hyperparameters_path", required=True, metavar="/path/to/hyperparameters/", help="Path of the hyperparameters that will be used for pre-training. Hyperparameters should be stored in a yaml file.")
args = parser.parse_args()

import pandas as pd

try:
    chembl_df = pd.read_csv(args.selfies_dataset)
except FileNotFoundError:
    print("SELFIES dataset was not found. SMILES dataset provided. Converting SMILES to SELFIES.")
    from prepare_pretraining_data import prepare_data

    prepare_data(path=args.smiles_dataset, save_to=args.selfies_dataset)
    chembl_df = pd.read_csv(args.selfies_dataset)
print("SELFIES .csv is ready.")

print("Creating SELFIES .txt for tokenization.")
from os.path import isfile  # returns True if the file exists else False.

if not isfile(args.prepared_data_path):
    from prepare_pretraining_data import create_selfies_file

    if args.subset_size != 0:
        create_selfies_file(chembl_df, subset_size=args.subset_size, do_subset=True, save_to=args.prepared_data_path)
    else:
        create_selfies_file(chembl_df, do_subset=False, save_to=args.prepared_data_path)
print("SELFIES .txt is ready for tokenization.")

print("Creating BPE tokenizer.")
if not isfile(args.bpe_path+"/merges.txt"):
    import bpe_tokenizer

    bpe_tokenizer.bpe_tokenizer(path=args.prepared_data_path, save_to=args.bpe_path)
print("BPE Tokenizer is ready.")

print("Creating RobertaTokenizerFast.")
if not isfile(args.roberta_fast_tokenizer_path+"/merges.txt"):
    import roberta_tokenizer
    
    roberta_tokenizer.save_roberta_tokenizer(path=args.bpe_path, save_to=args.roberta_fast_tokenizer_path)
print("RobertaFastTokenizer is ready.")

import yaml
import roberta_model

with open(args.hyperparameters_path) as file:
    hyperparameters = yaml.safe_load(file)
    for key in hyperparameters.keys():
        print("Starting pretraining with {} parameter set.".format(key))
        roberta_model.train_and_save_roberta_model(hyperparameters_dict=hyperparameters[key], selfies_path=args.prepared_data_path, robertatokenizer_path=args.roberta_fast_tokenizer_path, save_to="./saved_models/" + key + "_saved_model/")
        print("Finished pretraining with {} parameter set.\n---------------\n".format(key))
