import os
import numpy as np
import pandas as pd
import torch
from simpletransformers.classification import MultiLabelClassificationModel
from prepare_finetuning_data import smiles_to_selfies
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="sider", help="task selection.")
parser.add_argument("--pred_set", default="data/finetuning_datasets/classification/sider/sider_mock.csv", metavar="/path/to/dataset/", help="Test set for predictions.")
parser.add_argument("--training_args", default= "data/finetuned_models/SELFormer_sider_scaffold_optimized/training_args.bin", metavar="/path/to/dataset/", help="Trained model arguments.")
parser.add_argument("--model_name", default="data/finetuned_models/SELFormer_sider_scaffold_optimized",  metavar="/path/to/dataset/", help="Path to the model.")
parser.add_argument("--num_labels", default=27, type=int, help="Number of labels.")
args = parser.parse_args()

print("Loading test set...")
pred_set = pd.read_csv(args.pred_set)
pred_df_selfies = smiles_to_selfies(pred_set)

print("Loading model...")
training_args = torch.load(args.training_args)
num_labels = args.num_labels
model = MultiLabelClassificationModel("roberta", args.model_name, num_labels=num_labels, use_cuda=True, args=args.training_args)

print("Predicting...")
preds, _ = model.predict(pred_df_selfies["selfies"].tolist())

# create a dataframe with the selfies and the predictions each in a seperate column named feature_0, feature_1, etc.
res = pd.DataFrame(preds, columns=["feature_{}".format(i) for i in range(num_labels)])
res.insert(0, "selfies", pred_df_selfies["selfies"].tolist())

if not os.path.exists("data/predictions"):
    os.makedirs("data/predictions")

res.to_csv("data/predictions/{}_predictions.csv".format(args.task), index=False)
print("Predictions saved to data/predictions/{}_predictions.csv".format(args.task))