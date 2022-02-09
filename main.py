import pandas as pd
try:
	chembl_df = pd.read_csv("./data/chembl_29_selfies.csv")
	print("./data/chembl_29_selfies.csv already exists.")
except FileNotFoundError:
	print("creating ./data/chembl_29_selfies.csv")
	import prepare_data
	prepare_data.prepare_data(path="./data/chembl_29_chemreps.txt", save_to="./data/chembl_29_selfies.csv")
	chembl_df = pd.read_csv("./data/chembl_29_selfies.csv")
print("chembl_29_selfies is read.")

from os.path import isfile  # returns True if the file exists else False.
if not isfile("./data/chembl_29_selfies_alphabet.txt"):
	import create_selfies_alphabet
	create_selfies_alphabet.get_selfies_alphabet(chembl_df)
	print("created ./data/chembl_29_selfies_alphabet.txt")

if not isfile("./data/selfies_subset.txt"):
	print("creating ./data/selfies_subset.txt")
	import create_subset
	create_subset.create_selfies_subset(chembl_df, subset_size=100000, save_to="./data/selfies_subset.txt")
print("./data/selfies_subset.txt is available.")

if not isfile("./data/bpe/bpe.json"):
	print("creating ./data/bpe/bpe.json")
	import bpe_tokenizer
	bpe_tokenizer.bpe_tokenizer(path="./data/selfies_subset.txt", save_to="./data/bpe/")
print("./data/bpe/bpe.json is available.")

if not isfile("./data/robertatokenizer/merges.txt") or not isfile("./data/robertatokenizer/special_tokens_map.json") or not isfile("./data/robertatokenizer/tokenizer_config.json") or not isfile("./data/robertatokenizer/tokenizer.json") or not isfile("./data/robertatokenizer/vocab.json"):
	print("creating ./data/robertatokenizer/")
	import roberta_tokenizer
	roberta_tokenizer.save_roberta_tokenizer(path="./data/bpe/", save_to="./data/robertatokenizer/")
print("./data/robertatokenizer/ is available.")

import yaml
import roberta_model
import test_roberta_model
with open("hyperparameters.yml") as file:
	hyperparameters = yaml.safe_load(file)
	for key in hyperparameters.keys():
		print("starting pre-training with {} parameter set.".format(key))
		roberta_model.train_and_save_roberta_model(
			hyperparameters_dict=hyperparameters[key],
			selfies_path="./data/selfies_subset.txt",
			robertatokenizer_path="./data/robertatokenizer/",
			save_to="./"+key+"_saved_model/")
		test_roberta_model.test_roberta_model(model_folder="./"+key+"_saved_model/", roberta_tokenizer_folder="./data/robertatokenizer/")
