"""
TO RUN THIS FILE

python train_classification_model.py --dataset=data/finetuning/classification/hiv/hiv.csv --model=data/saved_models/molbert_100epochs_saved_model --save_to=molbert_hiv_model

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import (
    Dataset
)

from transformers import  (
    BertPreTrainedModel, 
    RobertaConfig, 
    RobertaTokenizerFast
)

from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)

# Model
class RobertaForSelfiesClassification(BertPreTrainedModel):
    
    def __init__(self, config):
        super(RobertaForSelfiesClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        
        
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.roberta(input_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


model_name = './data/saved_models/molbert_100epochs_saved_model'
tokenizer_name = './data/robertatokenizer'


# Configs
num_labels = 2
config_class = RobertaConfig
config = config_class.from_pretrained(model_name, num_labels=num_labels)

model_class = RobertaForSelfiesClassification
model = model_class.from_pretrained(model_name, config=config)

tokenizer_class = RobertaTokenizerFast
tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)


# Prepare and Get Data
class MyClassificationDataset(Dataset):
    
    def __init__(self, data, tokenizer, MAX_LEN):
        text, labels = data
        self.examples = tokenizer(text=text,text_pair=None,truncation=True,padding="max_length",
                                max_length=MAX_LEN,
                                return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        item = {key: self.examples[key][index] for key in self.examples}
        item['label'] = self.labels[index]
        return item


# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    metavar="/path/to/model",
                    help="Path to model")
parser.add_argument('--dataset', required=True,
                    metavar="/path/to/dataset/",
                    help='Directory of the dataset')
parser.add_argument('--save_to', required=True,
                    metavar="/path/to/save/to/",
                    help='Directory to save the model')
args = parser.parse_args()


DATASET_PATH = args.dataset
SPLIT_DATA_PATH = os.path.join(os.path.dirname(args.dataset), "dataframes")

# check if the directory for dataframes already exist
if not os.path.exists(SPLIT_DATA_PATH):
    # create a new directory
    os.makedirs(SPLIT_DATA_PATH)


from prepare_finetuning_data import smiles_to_selfies
from prepare_finetuning_data import train_val_test_split

# files = os.listdir(SPLIT_DATA_PATH)
# # check if the files already exist (csv files for train-validation-test)
# if "train_df.csv" in files and "validation_df.csv" in files and "test_df.csv" in files:
#     train_df = pd.read_csv(os.path.join(SPLIT_DATA_PATH, "train_df.csv"), sep=",")
#     validation_df = pd.read_csv(os.path.join(SPLIT_DATA_PATH, "validation_df.csv"), sep=",")
#     test_df = pd.read_csv(os.path.join(SPLIT_DATA_PATH, "test_df.csv"), sep=",")
#     test_y = pd.DataFrame(test_df.target, columns = ['target'])
# else:
#     # split data

(train, val, test) = train_val_test_split(DATASET_PATH)

train_smiles = [item[0] for item in train.smiles()]
validation_smiles = [item[0] for item in val.smiles()]
test_smiles = [item[0] for item in test.smiles()]

train_df = pd.DataFrame(np.column_stack([train_smiles, train.targets()]), columns = ['smiles', 'target'])
validation_df = pd.DataFrame(np.column_stack([validation_smiles, val.targets()]), columns = ['smiles', 'target'])
test_df = pd.DataFrame(np.column_stack([test_smiles, test.targets()]), columns = ['smiles', 'target'])

# convert dataframes to SELFIES representation
smiles_to_selfies(train_df, os.path.join(SPLIT_DATA_PATH,"train_df.csv"))
smiles_to_selfies(validation_df, os.path.join(SPLIT_DATA_PATH,"validation_df.csv"))
smiles_to_selfies(test_df, os.path.join(SPLIT_DATA_PATH,"test_df.csv"))

train_df = pd.read_csv(os.path.join(SPLIT_DATA_PATH, "train_df.csv"), sep=",")
validation_df = pd.read_csv(os.path.join(SPLIT_DATA_PATH, "validation_df.csv"), sep=",")
test_df = pd.read_csv(os.path.join(SPLIT_DATA_PATH, "test_df.csv"), sep=",")
test_y = pd.DataFrame(test_df.target, columns = ['target'])

MAX_LEN = 128 
train_examples = (train_df.iloc[:, 0].astype(str).tolist(), train_df.iloc[:, 1].tolist())
train_dataset = MyClassificationDataset(train_examples, tokenizer, MAX_LEN)

validation_examples = (validation_df.iloc[:, 0].astype(str).tolist(), validation_df.iloc[:, 1].tolist())
validation_dataset = MyClassificationDataset(validation_examples, tokenizer, MAX_LEN)

test_examples = (test_df.iloc[:, 0].astype(str).tolist(), test_df.iloc[:, 1].tolist())
test_dataset = MyClassificationDataset(test_examples, tokenizer, MAX_LEN)


from datasets import load_metric
acc = load_metric('accuracy')
precision = load_metric('precision')
recall = load_metric('recall')
f1 = load_metric('f1')

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc_result = acc.compute(predictions=predictions, references=labels)
    precision_result = precision.compute(predictions=predictions, references=labels)
    recall_result = recall.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels)

    result = {**acc_result, **precision_result, **recall_result, **f1_result}
    return result


from transformers import TrainingArguments, Trainer

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TRAIN_EPOCHS = 25
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
MAX_LEN = MAX_LEN

training_args = TrainingArguments(
    output_dir=args.save_to,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    save_total_limit=1,
    disable_tqdm=True,
    # fp16=True
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=validation_dataset,     # evaluation dataset
    compute_metrics=compute_metrics
)

metrics = trainer.train()
print("Metrics")
print(metrics)
trainer.save_model(args.save_to)

# Testing
# Make prediction
raw_pred, label_ids, metrics = trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

roc_auc_score_result = roc_auc_score(y_true = test_y, y_score= y_pred) 
# calculate precision-recall curve
precision_from_curve, recall_from_curve, thresholds_from_curve = precision_recall_curve(test_y, y_pred)
# calculate precision-recall AUC
auc_score = auc(recall_from_curve, precision_from_curve)
print(metrics)
print("\nROC-AUC: ",roc_auc_score_result,"\nPRC-AUC: ", auc_score)#, precision_from_curve, recall_from_curve)