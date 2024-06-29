import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
from pandarallel import pandarallel

from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch

df = pd.read_csv("./data/molecule_dataset_selfies.csv") # path of the selfies data

model_name = "./data/pretrained_models/SELFormer" # path of the pre-trained model
config = RobertaConfig.from_pretrained(model_name)
config.output_hidden_states = True
tokenizer = RobertaTokenizer.from_pretrained("./data/RobertaFastTokenizer")
model = RobertaModel.from_pretrained(model_name, config=config)


def get_sequence_embeddings(selfies):
    token = torch.tensor([tokenizer.encode(selfies, add_special_tokens=True, max_length=512, padding=True, truncation=True)])
    output = model(token)

    sequence_out = output[0]
    return torch.mean(sequence_out[0], dim=0).tolist()

print("Starting")
df = df[:100000] # how many molecules should be processed
pandarallel.initialize(nb_workers=5) # number of threads
df["sequence_embeddings"] = df.selfies.parallel_apply(get_sequence_embeddings)

df.drop(columns=["selfies"], inplace=True) # not interested in selfies data anymore, only chembl_id and the embedding
df.to_csv("./data/embeddings.csv", index=False) # save embeddings here
print("Finished")
