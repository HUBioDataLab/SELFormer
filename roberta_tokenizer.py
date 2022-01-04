from transformers import RobertaTokenizerFast


def save_roberta_tokenizer(pretrained_path="./data/bpe/", save_path="./data/robertatokenizer/"):
	tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_path)
	print("Loaded pretrained BPE tokenizer from: " + pretrained_path)

	tokenizer.save_pretrained(save_path)
	print("Saved RobertaTokenizerFast to: " + save_path)
