import pandas as pd
try:
	chembl_df = pd.read_csv("./data/chembl_29_selfies.csv")
except FileNotFoundError:
	import prepare_data
	prepare_data.prepare_data(path="data/chembl_29_chemreps.txt", save_to="./data/chembl_29_selfies.csv")
	chembl_df = pd.read_csv("./data/chembl_29_selfies.csv")


from os.path import isfile  # returns True if the file exists else False.
if not isfile("./data/chembl_29_selfies_alphabet.txt"):
	import create_selfies_alphabet
	create_selfies_alphabet.get_selfies_alphabet(chembl_df)


import create_subset
create_subset.create_selfies_subset(chembl_df, 100000, save_to="./data/selfies_subset.txt")


import bpe_tokenizer
bpe_tokenizer.bpe_tokenizer(path="./data/selfies_subset.txt", save_to="./data/bpe/")


import roberta_tokenizer
roberta_tokenizer.save_roberta_tokenizer(path="./data/bpe/", save_to="./data/robertatokenizer/")

import yaml
import roberta_model
with open("hyperparameters.yaml") as file:
	hyperparameters = yaml.safe_load(file)
	for key in hyperparameters.keys():
		roberta_model.train_and_save_roberta_model(hyperparameters_dict=hyperparameters[key], selfies_path="./data/selfies_subset.txt", bpe_path="./data/bpe/", save_to="./"+key+"_saved_model/")

import test_roberta_model
test_roberta_model.test_roberta_model(model_folder="./saved_model/", roberta_tokenizer_folder="./data/robertatokenizer/")
