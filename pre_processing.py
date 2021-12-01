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

with open("./data/chembl_29_selfies_alphabet.txt", "w") as f:
    f.write(",".join(list(selfies_alphabet)))

#

selfies_df = pd.read_csv("./data/chembl_29_selfies.csv")
selfies_df.sample(frac=1).reset_index(drop=True) # shuffling

#

selfies_subset = selfies_df.selfies[:100000]
selfies_subset = selfies_subset.to_frame()
selfies_subset["selfies"].to_csv("./data/selfies_only.txt", index=False)

#

from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

#

from tokenizers.pre_tokenizers import Split
from tokenizers import Regex

tokenizer.pre_tokenizer = Split(pattern=Regex("\[|\]"), behavior="removed")

#

from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[("<s>", 1), ("</s>", 2)],
)

#

from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
tokenizer.train(files=["./data/selfies_subset.txt"], trainer=trainer)

#

output = tokenizer.encode("[C][=C][C][=C][C][=C][Ring1][=Branch1]")
print(output.tokens)
print(output.ids)

#

tokenizer.save("./data/bpe/bpe.json", pretty=True)
tokenizer.model.save("./data/bpe/")