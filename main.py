import pandas as pd

try:
	chembl_df = pd.read_csv("./data/chembl_29_selfies.csv")
except FileNotFoundError:
	import prepare_data

	prepare_data.prepare_data()
	chembl_df = pd.read_csv("./data/chembl_29_selfies.csv")


from os.path import isfile  # returns True if the file exists else False.

if not isfile("./data/chembl_29_selfies_alphabet.txt"):
	import create_selfies_alphabet

	create_selfies_alphabet.get_selfies_alphabet(chembl_df)


import create_subset

create_subset.create_selfies_subset(chembl_df, 100000)


import bpe_tokenizer

bpe_tokenizer.bpe_tokenizer()


import roberta_tokenizer

roberta_tokenizer.save_roberta_tokenizer()

import yaml

with open("hyperparameters.yaml") as file:
	hyperparameters = yaml.load(file)
	print(hyperparameters)


import roberta_model

roberta_model.train_and_save_roberta_model()


import test_roberta_model

test_roberta_model.test_roberta_model()
