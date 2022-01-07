from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


def bpe_tokenizer(path="./data/selfies_subset.txt", save_to="./data/bpe/"):

	tokenizer = Tokenizer(BPE(unk_token="<unk>"))

	tokenizer.pre_tokenizer = Split(pattern=Regex("\[|\]"), behavior="removed")

	tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>", pair="<s> $A </s> $B:1 </s>:1", special_tokens=[("<s>", 1), ("</s>", 2)],)

	trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"])
	tokenizer.train(files=[path], trainer=trainer)

	tokenizer.save(save_to + "bpe.json", pretty=True)
	tokenizer.model.save(save_to)