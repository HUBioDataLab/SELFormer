import selfies as sf


def to_selfies(smiles):  # returns selfies representation of smiles string. if there is no representation return smiles unchanged.
    try:
        return sf.encoder(smiles)
    except sf.EncoderError:
        return smiles
