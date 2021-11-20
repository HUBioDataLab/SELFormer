import pandas as pd
import numpy as np

#

chembl_df = pd.read_csv("data/chembl_29_chemreps.txt", sep="\t")
chembl_df.drop(columns=["standard_inchi", "standard_inchi_key"], inplace=True)

chembl_df["selfies"] = chembl_df["canonical_smiles"]

#

import selfies as sf
def to_selfies(smiles): # returns selfies representation of smiles string. if there is no representation return smiles unchanged.
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        return smiles

#

from pandarallel import pandarallel
pandarallel.initialize()
chembl_df.selfies = chembl_df.selfies.parallel_apply(to_selfies)

#

chembl_df.drop(chembl_df[chembl_df.canonical_smiles == chembl_df.selfies].index, inplace=True)
chembl_df.to_csv("./data/chembl_29_selfies.csv", index=False)

#

selfies_array = chembl_df.selfies.to_numpy(copy=True)
selfies_alphabet = sf.get_alphabet_from_selfies(selfies_array)

#

with open("./data/chembl_29_selfies_alphabet.txt", "w") as f:
    f.write(",".join(list(selfies_alphabet)))

#

selfies_df = pd.read_csv("./data/chembl_29_selfies.csv")
selfies_df.sample(frac=1).reset_index(drop=True) # shuffling

#

def white_spaced_selfies(selfie_row): # takes a selfie string and returns it whitespaced. eg. '[C][O][H][H]' -> '[C] [O] [H] [H]'
    selfies_list = list(sf.split_selfies(selfie_row))
    return " ".join(i for i in selfies_list)

#

selfies_subset = selfies_df.selfies[:100000]
selfies_subset = selfies_subset.to_frame()

#

pandarallel.initialize()
selfies_subset["whitespaced_selfies"] = selfies_subset["selfies"].parallel_apply(white_spaced_selfies)

#

selfies_subset["selfies"].to_csv("./data/selfies_only.txt", index=False)
selfies_subset["whitespaced_selfies"].to_csv("./data/selfies_only_whitespaced.txt", index=False)

#

from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())

#

from tokenizers.pre_tokenizers import WhitespaceSplit

tokenizer.pre_tokenizer = WhitespaceSplit()

#

from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer()
tokenizer.train(files=["./data/selfies_only_whitespaced.txt"], trainer=trainer)
tokenizer.save("./data/bpe.json", pretty=True)