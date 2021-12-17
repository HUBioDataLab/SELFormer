from tokenizers import Tokenizer
from tokenizers.models import BPE

from tokenizers.pre_tokenizers import Split
from tokenizers import Regex

from tokenizers.processors import TemplateProcessing

from tokenizers.trainers import BpeTrainer

def bpe_tokenizer(path="./data/selfies_subset.txt"):

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    #tokenizer.pre_tokenizer = Split(pattern=Regex("\[|\]"), behavior="removed")
    tokenizer.pre_tokenizer = Split(pattern=Regex("\[.*?\]"), behavior="isolated")
    
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )

    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
    tokenizer.train(files=[path], trainer=trainer)

    output = tokenizer.encode("[C][=C][C][=C][C][=C][Ring1][=Branch1]")
    print(output.tokens)
    print(output.ids)

    tokenizer.save("./data/bpe/bpe.json", pretty=True)
    tokenizer.model.save("./data/bpe/")
