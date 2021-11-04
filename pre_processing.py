import pandas as pd

chembl_df = pd.read_csv("data/chembl_29_chemreps.txt", sep="\t")
chembl_df.drop(columns=["standard_inchi", "standard_inchi_key"], inplace=True)

chembl_df["selfies"] = chembl_df["canonical_smiles"]
chembl_df.head()

import selfies as sf
def to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        return smiles

from pandarallel import pandarallel
pandarallel.initialize()
chembl_df.selfies = chembl_df.selfies.parallel_apply(to_selfies)

chembl_df.drop(chembl_df[chembl_df.canonical_smiles == chembl_df.selfies].index, inplace=True)
chembl_df.to_csv("data/chembl_29_selfies.csv", index=False)

selfies_array = chembl_df.selfies.to_numpy(copy=True)
selfies_alphabet = sf.get_alphabet_from_selfies(selfies_array)
print(list(selfies_alphabet))
print(len(selfies_alphabet))

with open("data/chembl_29_selfies_alphabet.txt", "w") as f:
    f.write(",".join(list(selfies_alphabet)))