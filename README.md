# SELFormer: Molecular Representation Learning via SELFIES Language Models

<!-- omit in toc -->

[![publication](https://img.shields.io/badge/Article-%40MLST%20journal-d30000.svg)](https://doi.org/10.1088/2632-2153/acdb30) [![license](https://img.shields.io/badge/license-GPLv3-blue.svg)](http://www.gnu.org/licenses/)

Automated computational analysis of the vast chemical space is critical for numerous fields of research such as drug discovery and material science. Representation learning techniques have recently been employed with the primary objective of generating compact and informative numerical expressions of complex data. One approach to efficiently learn molecular representations is processing string-based notations of chemicals via natural language processing (NLP) algorithms. Majority of the methods proposed so far utilize SMILES notations for this purpose; however, SMILES is associated with numerous problems related to validity and robustness, which may prevent the model from effectively uncovering the knowledge hidden in the data. In this study, we propose SELFormer, a transformer architecture-based chemical language model that utilizes a 100% valid, compact and expressive notation, SELFIES, as input, in order to learn flexible and high-quality molecular representations. SELFormer is pre-trained on two million drug-like compounds and fine-tuned for diverse molecular property prediction tasks. Our performance evaluation has revealed that, SELFormer outperforms all competing methods, including graph learning-based approaches and SMILES-based chemical language models, on predicting aqueous solubility of molecules and adverse drug reactions. We also visualized molecular representations learned by SELFormer via dimensionality reduction, which indicated that even the pre-trained model can discriminate molecules with differing structural properties. We shared SELFormer as a programmatic tool, together with its datasets and pre-trained models. Overall, our research demonstrates the benefit of using the SELFIES notations in the context of chemical language modeling and opens up new possibilities for the design and discovery of novel drug candidates with desired features.

<img width="650" alt="Figure1_selformer_architecture" src="https://user-images.githubusercontent.com/13165170/229302081-94951d41-6f35-4f0f-a6dc-8c5914984f25.png">

**Figure.** The schematic representation of the SELFormer architecture and the experiments conducted. **Left:** the self-supervised pre-training utilizes the transformer encoder module via masked language modeling for learning concise and informative representations of small molecules encoded by their SELFIES notation. **Right:** the pre-trained model has been fine-tuned independently on numerous molecular property-based classification and regression tasks.


<br/>

## The Architecture of SELFormer

SELFormer is built on the RoBERTa transformer architecture, which utilizes the same architecture as BERT, but with certain modifications that have been found to improve model performance or provide other benefits. One such modification is the use of byte-level Byte-Pair Encoding (BPE) for tokenization instead of character-level BPE. Another one is that, RoBERTa is pre-trained exclusively on the masked language modeling (MLM) objective while disregarding the next sentence prediction (NSP) task. SELFormer has (i) self-supervised pre-trained models that utilize the transformer encoder module for learning concise and informative representations of small molecules encoded by their SELFIES notation, and (ii) supervised classification/regression models which use the pre-treined model as base and fine-tune on numerous classification- and regression-based molecular property prediction tasks.

Our pre-trained encoder models are implemented as "RobertaMaskedLM" and fine-tuning models as "RobertaForSequenceClassification". For the fine-tuning process, the SELFormer architecture includes the pre-trained RoBERTa model as its base, and "RobertaClassificationHead" class as the following layers (for classification and regression). "RobertaClassificationHead" class consists of a dropout layer, a dense layer, tanh activation function, a dropout layer, and a final linear layer. We forward the sequence output of the pre-trained RoBERTa base model to the classifier during the fine-tuning process.

<br/>

## Getting Started

We highly recommend the Conda platform for installing dependencies. Following the installation of Conda, please create and activate an environment with dependencies as defined below:

```
conda create -n SELFormer_env
conda activate SELFormer_env
conda env update --file data/requirements.yml
```
<br/>

## Generating Molecule Embeddings Using Pre-trained Models

Pre-trained SELFormer models are available for download [here](https://drive.google.com/drive/folders/1c3Mwc3j4M0PHk_iORrKU_V5cuxkD9aM6?usp=share_link). Embeddings of all molecules from CHEMBL30 and CHEMBL33 that are generated by our best performing model are available [here](https://drive.google.com/drive/folders/1Ii44Z6HonzJv5B5VYFujVaSThf802e2M?usp=sharing). 

You can also generate embeddings for your own dataset using the pre-trained models. To do so, you will need SELFIES notations of your molecules. You can use the command below to generate SELFIES notations for your SMILES dataset.

If you want to reproduce our code for generating embeddings of CHEMBL30 dataset, you can unzip __molecule_dataset_smiles.zip__ and/or __molecule_dataset_selfies.zip__ files in the __data__ directory and use them as input SMILES and SELFIES datasets, respectively.

```
python3 generate_selfies.py --smiles_dataset=data/molecule_dataset_smiles.txt --selfies_dataset=data/molecule_dataset_selfies.csv
```

* __--smiles_dataset__: Path of the input SMILES dataset.
* __--selfies_dataset__: Path of the output SELFIES dataset.

<br/>

To generate embeddings for the SELFIES molecule dataset using a pre-trained model, please run the following command:

```
python3 produce_embeddings.py --selfies_dataset=data/molecule_dataset_selfies.csv --model_file=data/pretrained_models/SELFormer --embed_file=data/embeddings.csv
```

* __--selfies_dataset__: Path of the input SELFIES dataset.
* __--model_file__: Path of the pretrained model to be used.
* __--embed_file__: Path of the output embeddings file.

<br/>

### Generating Embeddings Using Pre-trained Models for MoleculeNet Dataset Molecules

The embeddings generated by our best performing pre-trained model for MoleculeNet data can be directly downloaded [here](https://drive.google.com/drive/folders/1Xu3Q1T-KwXb67MF3Uw63pFm2IzoxeNNY?usp=share_link).

You can also re-generate these embeddings using the command below.

```
python3 get_moleculenet_embeddings.py --dataset_path=data/finetuning_datasets --model_file=data/pretrained_models/SELFormer 
```
* __--dataset_path__: Path of the directory containing the MoleculeNet datasets.
* __--model_file__: Path of the pretrained model to be used.

<br/>

## Training and Evaluating Models

### Pre-Training
To pre-train a model, please run the command below. If you have a SELFIES dataset, you can use it directly by giving the path of the dataset to __--selfies_dataset__. If you have a SMILES dataset, you can give the path of the dataset to __--smiles_dataset__ and the SELFIES representations will be created at the path given to __--selfies_dataset__.

<br/>

```
python3 train_pretraining_model.py --smiles_dataset=data/molecule_dataset_smiles.txt --selfies_dataset=data/molecule_dataset_selfies.csv --prepared_data_path=data/selfies_data.txt --bpe_path=data/BPETokenizer --roberta_fast_tokenizer_path=data/RobertaFastTokenizer --hyperparameters_path=data/pretraining_hyperparameters.yml --subset_size=100000
```

* __--smiles_dataset__: Path of the SMILES dataset. It is required if __--selfies_dataset__ does not exist (optional).
* __--selfies_dataset__: Path of the SELFIES dataset. If a SELFIES dataset does not exist, it will be created at the given path using the __--smiles_dataset__. If it exists, SELFIES dataset will be used directly (required).
* __--prepared_data_path__: Path of the intermediate file that will be created during pre-training. It will be used for tokenization. If it does not exist, it will be created at the given path (required).
* __--bpe_path__: Path of the BPE tokenizer. If it does not exist, it will be created at the given path (required).
* __--roberta_fast_tokenizer_path__: Path of the RobertaTokenizerFast tokenizer. If it does not exist, it will be created at the given path (required).
* __--hyperparameters_path__: Path of the yaml file that contains the hyperparameter sets to be tested. Note that these sets will be tested one by one and not in parallel. Example file is available at /data/pretraining_hyperparameters.yml (required).
* __--subset_size__: The size of the subset of the dataset that will be used for pre-training. By default, the whole dataset will be used (optional).

<br/>

### Fine-tuning on Molecular Property Prediction

You can use commands below to fine-tune a pre-trained model for various molecular property prediction tasks. These commands are utilized to handle datasets containing SMILES representations of molecules. SMILES representations should be stored in a column with a header named "smiles". You can see the example datasets in the __data/finetuning_datasets__ directory. 

<br/>

**Binary Classification Tasks**

To fine-tune a pre-trained model on a binary classification dataset, please run the command below. 

```
python3 train_classification_model.py --model=data/saved_models/SELFormer --tokenizer=data/RobertaFastTokenizer --dataset=data/finetuning_datasets/classification/bbbp/bbbp.csv --save_to=data/finetuned_models/SELFormer_bbbp_classification --target_column_id=1 --use_scaffold=1 --train_batch_size=16 --validation_batch_size=8 --num_epochs=25 --lr=5e-5 --wd=0
```

* __--model__: Directory of the pre-trained model (required).
* __--tokenizer__: Directory of the RobertaFastTokenizer (required).
* __--dataset__: Path of the fine-tuning dataset (required).
* __--save_to__: Directory where the fine-tuned model will be saved (required).
* __--target_column_id__: Default: 1. The column id of the target column in the fine-tuning dataset (optional).
* __--use_scaffold__: Default: 0. Determines whether to use scaffold splitting (1) or random splitting (0) (optional).
* __--train_batch_size__: Default: 8 (optional).
* __--validation_batch_size__ : Default: 8 (optional).
* __--num_epochs__: Default: 50. Number of epochs (optional).
* __--lr__: Default: 1e-5: Learning rate (optional).
* __--wd__: Default: 0.1: Weight decay (optional).

<br/>

**Multi-Label Classification Tasks**

To fine-tune a pre-trained model on a multi-label classification dataset, please run the command below. The RobertaFastTokenizer files should be stored in the same directory as the pre-trained model.

```
python3 train_classification_multilabel_model.py --model=data/saved_models/SELFormer --dataset=data/finetuning_datasets/classification/tox21/tox21.csv --save_to=data/finetuned_models/SELFormer_tox21_classification --use_scaffold=1 --batch_size=16 --num_epochs=25 --lr=5e-5 --wd=0
```

* __--model__: Directory of the pre-trained model (required).
* __--dataset__: Path of the fine-tuning dataset (required).
* __--save_to__: Directory where the fine-tuned model will be saved (required).
* __--use_scaffold__: Default: 0. Determines whether to use scaffold splitting (1) or random splitting (0) (optional).
* __--batch_size__: Default: 8. Train batch size (optional).
* __--num_epochs__: Default: 50. Number of epochs (optional).
* __--lr__: Default: 1e-5: Learning rate (optional).
* __--wd__: Default: 0.1: Weight decay (optional).

<br/>

**Regression Tasks**

To fine-tune a pre-trained model on a regression dataset, please run the command below. 

```
python3 train_regression_model.py --model=data/saved_models/SELFormer --tokenizer=data/RobertaFastTokenizer --dataset=data/finetuning_datasets/regression/esol/esol.csv --save_to=data/finetuned_models/SELFormer_esol_regression --target_column_id=-1 --scaler=2 --use_scaffold=1 --train_batch_size=16 --validation_batch_size=8 --num_epochs=25 --lr=5e-5 --wd=0
```
 
* __--model__: Directory of the pre-trained model (required).
* __--tokenizer__: Directory of the RobertaFastTokenizer (required).
* __--dataset__: Path of the fine-tuning dataset (required).
* __--save_to__: Directory where the fine-tuned model will be saved (required).
* __--target_column_id__: Default: 1. The column id of the target column in the fine-tuning dataset (optional).
* __--scaler__: Default: 0. Method to be used for scaling the target values. 0 for no scaling, 1 for min-max scaling, 2 for standard scaling (optional).
* __--use_scaffold__: Default: 0. Determines whether to use scaffold splitting (1) or random splitting (0) (optional).
* __--train_batch_size__: Default: 8 (optional).
* __--validation_batch_size__ : Default: 8 (optional).
* __--num_epochs__: Default: 50. Number of epochs (optional).
* __--lr__: Default: 1e-5: Learning rate (optional).
* __--wd__: Default: 0.1: Weight decay (optional).

<br/>

## Producing Molecular Property Predictions with Fine-tuned Models 

Fine-tuned SELFormer models are available for download [here](https://drive.google.com/drive/folders/1LVw1YZBL1AUAGCxIkavz0KMJNVyzxAXG?usp=share_link). To make predictions with these models, please follow the instructions below.

<br/>

### Binary Classification

To make predictions for either BACE, BBBP, and HIV datasets, please run the command below. Change the indicated arguments for different tasks. Default parameters will load fine-tuned model on BBBP. 

```
python3 binary_class_pred.py --task=bbbp --model_name=data/finetuned_models/SELFormer_bbbp_scaffold_optimized --tokenizer=data/RobertaFastTokenizer --pred_set=data/finetuning_datasets/classification/bbbp/bbbp_mock.csv --training_args=data/finetuned_models/SELFormer_bbbp_scaffold_optimized/training_args.bin
```

* __--task__: Binary classification task to choose. (bace, bbbp, hiv) (required).
* __--model_name__: Path of the fine-tuned model (required).
* __--tokenizer__: Tokenizer selection (required).
* __--pred_set__: Molecules to make predictions. Should be a CSV file with a single column. Header should be smiles (required).
* __--training_args__: Initialize the model arguments (required).

<br/>

### Multi-Label Classification

To make predictions for either Tox21 and SIDER datasets, please run the command below. Change the indicated arguments for different tasks. Default parameters will load fine-tuned model on SIDER. 

```
python3 multilabel_class_pred.py --task=sider --model_name=data/finetuned_models/SELFormer_sider_scaffold_optimized --pred_set=data/finetuning_datasets/classification/sider/sider_mock.csv --training_args=data/finetuned_models/SELFormer_sider_scaffold_optimized/training_args.bin --num_labels=27
```

* __--task__: Multi-label classification task to choose. (tox21, sider) (required).
* __--model_name__: Path of the fine-tuned model (required).
* __--pred_set__: Molecules to make predictions. Should be a CSV file with a single column containing SMILES. Header should be 'smiles' (required).
* __--training_args__: Initialize the model arguments (required).
* __--num_labels__: Number of labels (required).

<br/>

### Regression

To make predictions for either ESOL, FreeSolv, Lipophilicity, and PDBBind datasets, please run the command below. Change the indicated arguments for different tasks. Default parameters will load fine-tuned model on ESOL. 

```
python3 regression_pred.py --task=esol --model_name=data/finetuned_models/esol_regression --tokenizer=data/RobertaFastTokenizer --pred_set=data/finetuning_datasets/classification/esol/esol_mock.csv --training_args=data/finetuned_models/esol_regression/training_args.bin 
```

* __--task__: Binary classification task to choose. (esol, freesolv, lipo, pdbbind_full) (required).
* __--model_name__: Path of the fine-tuned model (required).
* __--tokenizer__: Tokenizer selection (required).
* __--pred_set__: Molecules to make predictions. Should be a CSV file with a single column. Header should be smiles (required).
* __--training_args__: Initialize the model arguments (required).

<br/>

## License
Copyright (C) 2023 HUBioDataLab

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

