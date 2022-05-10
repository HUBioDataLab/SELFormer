import os

os.environ["TOKENIZER_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

from transformers import BertPreTrainedModel, RobertaConfig, RobertaTokenizerFast

from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)


# Parse command line arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, metavar="/path/to/model", help="Directory of the pre-trained model")
parser.add_argument("--tokenizer", required=True, metavar="/path/to/tokenizer/", help="Directory of the RobertaFastTokenizer")
parser.add_argument("--dataset", required=True, metavar="/path/to/dataset/", help="Path of the fine-tuning dataset")
parser.add_argument("--save_to", required=True, metavar="/path/to/save/to/", help="Directory to save the model")
parser.add_argument("--target_column_id", required=False, default="1", metavar="<int>", type=int, help="Column's ID in the dataframe")
parser.add_argument(
    "--use_scaffold", required=False, metavar="<int>", type=int, default=0, help="Split to use. 0 for random, 1 for scaffold. Default: 0",
)
parser.add_argument("--train_batch_size", required=False, metavar="<int>", type=int, default=8, help="Batch size for training. Default: 8")
parser.add_argument("--validation_batch_size", required=False, metavar="<int>", type=int, default=8, help="Batch size for validation. Default: 8")
parser.add_argument("--num_epochs", required=False, metavar="<int>", type=int, default=50, help="Number of epochs. Default: 50")
parser.add_argument("--lr", required=False, metavar="<float>", type=float, default=1e-5, help="Learning rate. Default: 1e-5")
parser.add_argument("--wd", required=False, metavar="<float>", type=float, default=0.1, help="Weight decay. Default: 0.1")
args = parser.parse_args()


# Model
class SELFIESTransformers_For_Classification(BertPreTrainedModel):
    def __init__(self, config):
        super(SELFIESTransformers_For_Classification, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


model_name = args.model
tokenizer_name = args.tokenizer


# Configs
num_labels = 2
config_class = RobertaConfig
config = config_class.from_pretrained(model_name, num_labels=num_labels)

model_class = SELFIESTransformers_For_Classification
model = model_class.from_pretrained(model_name, config=config)

tokenizer_class = RobertaTokenizerFast
tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)


# Prepare and Get Data
class SELFIESTransfomers_Dataset(Dataset):
    def __init__(self, data, tokenizer, MAX_LEN):
        text, labels = data
        self.examples = tokenizer(text=text, text_pair=None, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        item = {key: self.examples[key][index] for key in self.examples}
        item["label"] = self.labels[index]
        return item


DATASET_PATH = args.dataset
from prepare_finetuning_data import smiles_to_selfies
from prepare_finetuning_data import train_val_test_split

if args.use_scaffold == 0:  # random split
    print("Using random split")
    (train_df, validation_df, test_df) = train_val_test_split(DATASET_PATH, args.target_column_id, scaffold_split=False)
else:  # scaffold split
    print("Using scaffold split")
    (train, val, test) = train_val_test_split(DATASET_PATH, args.target_column_id, scaffold_split=True)

    train_smiles = [item[0] for item in train.smiles()]
    validation_smiles = [item[0] for item in val.smiles()]
    test_smiles = [item[0] for item in test.smiles()]

    train_df = pd.DataFrame(np.column_stack([train_smiles, train.targets()]), columns=["smiles", "target"])
    validation_df = pd.DataFrame(np.column_stack([validation_smiles, val.targets()]), columns=["smiles", "target"])
    test_df = pd.DataFrame(np.column_stack([test_smiles, test.targets()]), columns=["smiles", "target"])

train_df = smiles_to_selfies(train_df)
validation_df = smiles_to_selfies(validation_df)
test_df = smiles_to_selfies(test_df)
test_y = pd.DataFrame(test_df.target, columns=["target"])

MAX_LEN = 128
train_examples = (train_df.iloc[:, 0].astype(str).tolist(), train_df.iloc[:, 1].tolist())
train_dataset = SELFIESTransfomers_Dataset(train_examples, tokenizer, MAX_LEN)

validation_examples = (validation_df.iloc[:, 0].astype(str).tolist(), validation_df.iloc[:, 1].tolist())
validation_dataset = SELFIESTransfomers_Dataset(validation_examples, tokenizer, MAX_LEN)

test_examples = (test_df.iloc[:, 0].astype(str).tolist(), test_df.iloc[:, 1].tolist())
test_dataset = SELFIESTransfomers_Dataset(test_examples, tokenizer, MAX_LEN)


from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from datasets import load_metric

acc = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")
f1 = load_metric("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc_result = acc.compute(predictions=predictions, references=labels)
    precision_result = precision.compute(predictions=predictions, references=labels)
    recall_result = recall.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels)
    roc_auc_result = {"roc-auc": roc_auc_score(y_true=labels, y_score=predictions)}
    precision_from_curve, recall_from_curve, thresholds_from_curve = precision_recall_curve(labels, predictions)
    prc_auc_result = {"prc-auc": auc(recall_from_curve, precision_from_curve)}

    result = {**acc_result, **precision_result, **recall_result, **f1_result, **roc_auc_result, **prc_auc_result}
    return result


# Train and Evaluate
from transformers import TrainingArguments, Trainer

TRAIN_BATCH_SIZE = args.train_batch_size
VALID_BATCH_SIZE = args.validation_batch_size
TRAIN_EPOCHS = args.num_epochs
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.wd
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
    disable_tqdm=True,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="roc-auc",
    #     greater_is_better=True,
    save_total_limit=1,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=validation_dataset, compute_metrics=compute_metrics)  # the instantiated 🤗 Transformers model to be trained  # training arguments, defined above  # training dataset  # evaluation dataset

metrics = trainer.train()
print("Metrics")
print(metrics)
trainer.save_model(args.save_to)

# Testing
# Make prediction
raw_pred, label_ids, metrics = trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)

# ROC-AUC
roc_auc_score_result = roc_auc_score(y_true=test_y, y_score=y_pred)
# PRC-AUC
precision_from_curve, recall_from_curve, thresholds_from_curve = precision_recall_curve(test_y, y_pred)
auc_score_result = auc(recall_from_curve, precision_from_curve)

print("\nROC-AUC: ", roc_auc_score_result, "\nPRC-AUC: ", auc_score_result)
