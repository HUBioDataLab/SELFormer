# SELFIES-Transformer: Learning the Representation of Chemical Space for Discovering New Drugs using Transformers Architecture
Chemical and protein text representations can be conceived of as unstructured languages that humans have codified to describe domain-specific information. The use of natural language processing (NLP) to reveal that hidden knowledge in textual representations of these biological entities is at an all-time high. Discovering and developing new drugs is a critical topic considering the fast-growing and aging population and health risks caused by it, such as complex diseases (e.g., types of cancer). Conventional experimental procedures used for developing drugs are expensive, time-consuming and labor-intensive, which in turn decreases the efficiency of this process. In our paper, we proposed an NLP model that uses a large-scale pre-training methodology on 2 million molecules in their SELFIES representation to learn flexible and high-quality molecular representations for drug discovery problems, followed by a fine-tuning process to support varied downstream molecular analysis tasks such as molecular property prediction. As a result, our model outperformed [ChemBERTa](https://arxiv.org/abs/2010.09885) on all molecular analysis tasks and was only marginally behind [MolBERT](https://arxiv.org/abs/2011.13230) with good model interpretation ability. We hope that our strategy will reduce costs in the bioinformatics field, allowing researchers to continue their research without the need for additional funding.

## Installation
SELFIES-Transformer is a command-line tool for pre-training and fine-tuning using SELFIES represented molecules. It should run in any Unix-like operating system. You can install it using conda with commands below. [requirements.yml](/data/requirements.yml) is located under data folder.
```
conda create -n selfiesTransfomers_env
conda activate selfiesTransformers_env
conda env update --file data/requirements.yml
```
## The Transformer Architecture
Our pre-trained model is implemented as RobertaMaskedLM. We then achieve sequence outputs as the model's output to use for molecule representations. These representations will be used for fine-tuning. In order to use the sequence output for visualisation, we will be taking average of the sequence output.

Our fine-tuning model’s architecture was based on RobertaForSequenceClassification’s architecture. Our model’s architecture for the fine-tuning process includes a pre-trained RoBERTa model as a base model and RobertaClassificationHead class for the next layers as a classifier. RobertaClassificationHead class consists of a dropout layer, a dense layer, tanh activation function, a dropout layer, and a final linear layer for a classification task or a regression task in this order respectively. We forward the sequence output of the pre-trained RoBERTa base model to the classifier to use during the fine-tuning process for supervised tasks. We can achieve sequence outputs as the fine-tuned models's output for molecule representations. In order to use the sequence output for visualisation, again we will be taking average of the sequence output.

## Usage
### Pre-Training
You can use SELFIES-Transformer for pretraining task using either SMILES or SELFIES data.

```
python3 train_pretraining_model.py --smiles_dataset=data/chembl_29_chemreps.txt --selfies_dataset=data/chembl_29_selfies.csv --subset_size=100000 --prepared_data_path=data/selfies_data.txt --bpe_path=data/BPETokenizer --roberta_fast_tokenizer_path=data/RobertaFastTokenizer --hyperparameters_path=data/pretraining_hyperparameters.yml
```

* __--smiles_dataset__: Path of the SMILES dataset. If the dataset provided with __--selfies_dataset__ exists, then this argument is not required. Else, it is required.
* __--selfies_dataset__: Path of the SELFIES dataset. If it does not exist, it requires __--smiles_dataset__ argument and then it will be created at the given path. Required.
* __--subset_size__: By default the program will use the whole data. If you want to instead use a subset of the data, set this parameter to the size of the subset. Optional.
* __--prepared_data_path__: The file provided with __--selfies_dataset__ goes through some preprocessing and stored as a .txt file. If it does not exist, it will be created at the given path. Required.
* __--bpe_path__: Path of the BPE tokenizer. If it does not exist, it will be created at the given path. Required.
* __--roberta_fast_tokenizer_path__: Directory of the RobertaTokenizerFast tokenizer. RobertaFastTokenizer only depends on the BPE Tokenizer and will be created regardless of whether it exists or not. If the BPE Tokenizer did not change, this tokenizer will be the same as well. Required.
* __--hyperparameters_path__: Path of the hyperparameters that will be used for pre-training. It may contain multiple hyperparameters sets for multiple pre-training tasks. Note that these tasks will follow each other and not work simultaneously. Hyperparameters should be stored in a yaml file. Example file [pretraining_hyperparameters.yml](/data/pretraining_hyperparameters.yml) is under data folder. Required.

### Binary Classification
You can use the pre-trained models you trained and fine-tune them for binary classification tasks using SMILES data. Our program will convert it to SELFIES and train from there.

```
python3 train_classification_model.py --model=data/saved_models/modelO --tokenizer=data/RobertaFastTokenizer --dataset=data/finetuning_datasets/classification/bbbp/bbbp.csv --save_to=data/finetuned_models/modelO_bbbp_classification --target_column_id=1 --use_scaffold=1 --train_batch_size=16 --validation_batch_size=8 --num_epochs=25 --lr=5e-5 --wd=0
```

* __--model__: Directory of the pre-trained model. Required.
* __--tokenizer__: Directory of the RobertaFastTokenizer. Required.
* __--dataset__: Path of the fine-tuning dataset. Required.
* __--save_to__: Directory to save the model. Required.
* __--target_column_id__: Default: 1. By default the program assumes the target column is the second column of the dataframe. If this is not the case, set this value to column's number. Optional.
* __--use_scaffold__: Default: 0. By default the program will do random split on the fine-tuning dataset. Setting this value to 1 will cause the program to do scaffold split instead. Optional.
* __--train_batch_size__: Default: 8. Batch size for training. Optional.
* __--validation_batch_size__ : Default: 8. Batch size for validation. Optional.
* __--num_epochs__: Default: 50. Number of epochs to train. Optional.
* __--lr__: Default: 1e-5: Learning rate for fine-tuning.
* __--wd__: Default: 0.1: Weight decat for fine-tuning.

### Multi-Label Classification
You can use the pre-trained models you trained and fine-tune them for multi-label classification using SMILES data. Your RobertaFastTokenizer files need to be inside the folder provided by __--model__. Our program will convert it to SELFIES and train from there.

```
python3 train_classification_multilabel_model.py --model=data/saved_models/modelO --dataset=data/finetuning_datasets/classification/tox21/tox21.csv --save_to=data/finetuned_models/modelO_tox21_classification --use_scaffold=1 --batch_size=16 --num_epochs=25 --lr=5e-5 --wd=0
```

* __--model__: Directory of the pre-trained model. Required.
* __--dataset__: Path of the fine-tuning dataset. Required.
* __--save_to__: Directory to save the model. Required.
* __--use_scaffold__: Default: 0. By default the program will do random split on the fine-tuning dataset. Setting this value to 1 will cause the program to do scaffold split instead. Optional.
* __--batch_size__: Default: 8. Batch size for training. Optional.
* __--num_epochs__: Default: 50. Number of epochs to train. Optional.
* __--lr__: Default: 1e-5: Learning rate for fine-tuning.
* __--wd__: Default: 0.1: Weight decat for fine-tuning.

### Regression
You can use the pre-trained models you trained and fine-tune them for regression tasks using SMILES data. Our program will convert it to SELFIES and train from there.

```
python3 train_classification_model.py --model=data/saved_models/modelO --tokenizer=data/RobertaFastTokenizer --dataset=data/finetuning_datasets/classification/bbbp/bbbp.csv --save_to=data/finetuned_models/modelO_bbbp_classification --target_column_id=1 --scaler=2 --use_scaffold=1 --train_batch_size=16 --validation_batch_size=8 --num_epochs=25 --lr=5e-5 --wd=0
```
 
* __--model__: Directory of the pre-trained model. Required.
* __--tokenizer__: Directory of the RobertaFastTokenizer. Required.
* __--dataset__: Path of the fine-tuning dataset. Required.
* __--save_to__: Directory to save the model. Required.
* __--target_column_id__: Default: 1. By default the program assumes the target column is the second column of the dataframe. If this is not the case, set this value to column's number. Optional.
* __--scaler__: Default: 0. Scaler to use for regression. 0 for no scaling, 1 for min-max scaling, 2 for standard scaling. Optional.
* __--use_scaffold__: Default: 0. By default the program will do random split on the fine-tuning dataset. Setting this value to 1 will cause the program to do scaffold split instead. Optional.
* __--train_batch_size__: Default: 8. Batch size for training. Optional.
* __--validation_batch_size__ : Default: 8. Batch size for validation. Optional.
* __--num_epochs__: Default: 50. Number of epochs to train. Optional.
* __--lr__: Default: 1e-5: Learning rate for fine-tuning.
* __--wd__: Default: 0.1: Weight decat for fine-tuning.
