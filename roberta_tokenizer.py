from transformers import RobertaTokenizerFast


def save_roberta_tokenizer(path="./data/bpe/", save_to="./data/robertatokenizer/"):
	tokenizer = RobertaTokenizerFast.from_pretrained(path)
	print("Loaded pretrained BPE tokenizer from: " + path)

	tokenizer.save_pretrained(save_to)
	print("Saved RobertaTokenizerFast to: " + save_to)
