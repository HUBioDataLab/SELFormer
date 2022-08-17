
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--smiles_dataset", required=True, metavar="/path/to/dataset/", help="Path of the input SMILES dataset.")
parser.add_argument("--selfies_dataset", required=True, metavar="/path/to/dataset/", help="Path of the output SEFLIES dataset.")
args = parser.parse_args()

import pandas as pd
from prepare_pretraining_data import prepare_data

prepare_data(path=args.smiles_dataset, save_to=args.selfies_dataset)
chembl_df = pd.read_csv(args.selfies_dataset)
print("SELFIES representation file is ready.")