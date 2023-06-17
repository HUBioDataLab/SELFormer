import os
import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, RobertaConfig, RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)
from transformers import Trainer, TrainingArguments
from prepare_finetuning_data import smiles_to_selfies
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="esol", help="task selection.")
parser.add_argument("--tokenizer_name", default="data/RobertaFastTokenizer", metavar="/path/to/dataset/", help="Tokenizer selection.")
parser.add_argument("--pred_set", default='data/finetuning_datasets/regression/esol/esol_mock.csv', metavar="/path/to/dataset/", help="Test set for predictions.")
parser.add_argument("--training_args", default= "data/finetuned_models/esol_regression/training_args.bin", metavar="/path/to/dataset/", help="Trained model arguments.")
parser.add_argument("--model_name", default='data/finetuned_models/esol_regression',  metavar="/path/to/dataset/", help="Path to the model.")
args = parser.parse_args()

class SELFIESTransformers_For_Regression(BertPreTrainedModel):
    def __init__(self, config):
        super(SELFIESTransformers_For_Regression, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:  # regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
                outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

model_class = SELFIESTransformers_For_Regression

model_name = args.model_name
tokenizer_name = args.tokenizer_name
num_labels = 1
config_class = RobertaConfig
config = config_class.from_pretrained(model_name, num_labels=num_labels)

model_class = SELFIESTransformers_For_Regression
model = model_class.from_pretrained(model_name, config=config)

tokenizer_class = RobertaTokenizerFast
tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)

class SELFIESTransfomers_Dataset(Dataset):
    def __init__(self, data, tokenizer, MAX_LEN):
        text = data
        self.examples = tokenizer(text=text, text_pair=None, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
        

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, index):
        item = {key: self.examples[key][index] for key in self.examples}
        
        return item
    
pred_set = pd.read_csv(args.pred_set)
pred_df_selfies = smiles_to_selfies(pred_set)

MAX_LEN = 128

pred_examples = (pred_df_selfies.iloc[:, 0].astype(str).tolist())
pred_dataset = SELFIESTransfomers_Dataset(pred_examples, tokenizer, MAX_LEN)

training_args = torch.load(args.training_args)

trainer = Trainer(model=model, args=training_args)  # the instantiated 🤗 Transformers model to be trained  # training arguments, defined above  # training dataset  # evaluation dataset

raw_pred, label_ids, metrics = trainer.predict(pred_dataset)
y_pred = [i[0] for i in raw_pred]

res = pd.concat([pred_df_selfies, pd.DataFrame(y_pred, columns=["prediction"])], axis = 1)

if not os.path.exists("data/predictions"):
    os.makedirs("data/predictions")

res.to_csv("data/predictions/{}_predictions.csv".format(args.task), index=False)
