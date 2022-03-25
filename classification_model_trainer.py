import os
import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import (
    Dataset
)

import math 
from transformers import  (
    BertPreTrainedModel, 
    RobertaConfig, 
    RobertaTokenizerFast
)


from scipy.special import softmax
from torch.nn import CrossEntropyLoss, MSELoss


from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaConfig,
    RobertaModel,
)


save_to="./saved_classification_model/"
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
            if self.num_labels == 1: #regression
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int): #single label classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


model_name = './chemberta_saved_model/model'
num_labels = 2 #set it to 1 for regression
#device = torch.device("cuda")
tokenizer_name = './robertatokenizer'

# Configs

config_class = RobertaConfig
model_class = RobertaForSelfiesClassification
tokenizer_class = RobertaTokenizerFast

config = config_class.from_pretrained(model_name, num_labels=num_labels)

model = model_class.from_pretrained(model_name, config=config)
#print('Model=\n',model,'\n')

tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
#print('Tokenizer=',tokenizer,'\n')

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

train_df = pd.read_csv("ATC/ATC.csv", delimiter=";")[:15]
validation_df = pd.read_csv("ATC/ATC.csv", delimiter=";")[16:26]
test_df = pd.read_csv("ATC/ATC.csv", delimiter=";")[27:37]
test_y = pd.read_csv("ATC/ATC.csv", delimiter=";")["Label"][27:37]

MAX_LEN = 128 
train_examples = (train_df.iloc[:, 0].astype(str).tolist(), train_df.iloc[:, 1].tolist())
train_dataset = MyClassificationDataset(train_examples,tokenizer, MAX_LEN)

validation_examples = (validation_df.iloc[:, 0].astype(str).tolist(), validation_df.iloc[:, 1].tolist())
validation_dataset = MyClassificationDataset(validation_examples,tokenizer, MAX_LEN)

test_examples = (test_df.iloc[:, 0].astype(str).tolist(), test_df.iloc[:, 1].tolist())
test_dataset = MyClassificationDataset(test_examples,tokenizer, MAX_LEN)


# Train and Evaluate
from transformers import TrainingArguments, Trainer

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

import numpy as np

from datasets import load_metric
acc = load_metric('accuracy')
precision = load_metric('precision')
recall = load_metric('recall')
f1 = load_metric('f1')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc_result = acc.compute(predictions=predictions, references=labels)
    precision_result = precision.compute(predictions=predictions, references=labels)
    recall_result = recall.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels)

    result = {**acc_result, **precision_result, **recall_result, **f1_result}
    return result

TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
TRAIN_EPOCHS = 5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.1
MAX_LEN = MAX_LEN

training_args = TrainingArguments(
    output_dir=save_to + model_name,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    save_steps=8192,
    eval_steps=4096,
    save_total_limit=1
    #load_best_model_at_end=True,
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
trainer.save_model(save_to)

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
print("\nROC-AUC: ",roc_auc_score_result,"\n PRC-AUC: ", auc_score)#, precision_from_curve,recall_from_curve)