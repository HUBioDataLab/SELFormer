from transformers import pipeline


def test_roberta_model(model_folder="./saved_model/", roberta_tokenizer_folder="./data/robertatokenizer/", masked_selfies="<mask>[C][=C][C][Branch1][S][C][=C][S][C][Branch1][#Branch1][N][=C][Branch1][C][N][N][=N][Ring1][=Branch2][=C][N][Ring1][=C][C]"):
	
	fill_mask = pipeline("fill-mask", model=model_folder, tokenizer=roberta_tokenizer_folder)
	print(fill_mask(masked_selfies))
