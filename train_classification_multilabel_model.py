import os

os.environ["TOKENIZER_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

from simpletransformers.classification import MultiLabelClassificationModel

import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, metavar="/path/to/model", help="Path to model")
parser.add_argument("--dataset", required=True, metavar="/path/to/dataset/", help="Directory of the dataset")
parser.add_argument("--save_to", required=True, metavar="/path/to/save/to/", help="Directory to save the model")
parser.add_argument("--use_scaffold", required=False, metavar="<int>", type=int, default=0, help="Split to use. 0 for random, 1 for scaffold. Default: 0")
parser.add_argument("--num_epochs", required=False, metavar="<int>", type=int, default=50, help="Number of epochs. Default: 50")
parser.add_argument("--lr", required=False, metavar="<float>", type=float, default=1e-5, help="Learning rate. Default: 1e-5")
parser.add_argument("--wd", required=False, metavar="<float>", type=float, default=0.1, help="Weight decay. Default: 0.1")
parser.add_argument("--batch_size", required=False, metavar="<int>", type=int, default=8, help="Batch size. Default: 8")
args = parser.parse_args()


num_labels = len(pd.read_csv(args.dataset).columns) - 1
model_args = {
    "num_train_epochs": args.num_epochs,
    "learning_rate": args.lr,
    "weight_decay": args.wd,
    "train_batch_size": args.batch_size,
    "output_dir": args.save_to,
}

model = MultiLabelClassificationModel("roberta", args.model, num_labels=num_labels, use_cuda=True, args=model_args)

from prepare_finetuning_data import train_val_test_split_multilabel

if args.use_scaffold == 0:  # random split
    print("Using random split")
    (train_df, eval_df, test_df) = train_val_test_split_multilabel(args.dataset, scaffold_split=False)

    train_df.columns = ["smiles"] + ["Feature_" + str(i) for i in range(num_labels)]
    eval_df.columns = ["smiles"] + ["Feature_" + str(i) for i in range(num_labels)]
    test_df.columns = ["smiles"] + ["Feature_" + str(i) for i in range(num_labels)]
else:  # scaffold split
    print("Using scaffold split")
    (train, val, test) = train_val_test_split_multilabel(args.dataset, scaffold_split=True)

    train_smiles = [item[0] for item in train.smiles()]
    validation_smiles = [item[0] for item in val.smiles()]
    test_smiles = [item[0] for item in test.smiles()]

    train_df = pd.DataFrame(np.column_stack([train_smiles, train.targets()]), columns=["smiles"] + ["Feature_" + str(i) for i in range(len(train.targets()[0]))])
    eval_df = pd.DataFrame(np.column_stack([validation_smiles, val.targets()]), columns=["smiles"] + ["Feature_" + str(i) for i in range(len(val.targets()[0]))])
    test_df = pd.DataFrame(np.column_stack([test_smiles, test.targets()]), columns=["smiles"] + ["Feature_" + str(i) for i in range(len(test.targets()[0]))])

from prepare_finetuning_data import smiles_to_selfies

train_df = smiles_to_selfies(train_df)
eval_df = smiles_to_selfies(eval_df)
test_df = smiles_to_selfies(test_df)

train_df.insert(1, "labels", np.array([train_df["Feature_" + str(i)].to_numpy() for i in range(len(train_df.columns[1:]))], dtype=np.float32).T.tolist())
eval_df.insert(1, "labels", np.array([eval_df["Feature_" + str(i)].to_numpy() for i in range(len(eval_df.columns[1:]))], dtype=np.float32).T.tolist())
test_df.insert(1, "labels", np.array([test_df["Feature_" + str(i)].to_numpy() for i in range(len(test_df.columns[1:]))], dtype=np.float32).T.tolist())

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

from datasets import load_metric

acc = load_metric("accuracy")
precision = load_metric("precision")
recall = load_metric("recall")
f1 = load_metric("f1")


def compute_metrics(y_true, y_pred):
    acc_result = acc.compute(predictions=y_pred, references=y_true)
    precision_result = precision.compute(predictions=y_pred, references=y_true)
    recall_result = recall.compute(predictions=y_pred, references=y_true)
    f1_result = f1.compute(predictions=y_pred, references=y_true)
    roc_auc_result = {"roc-auc": roc_auc_score(y_true=y_true, y_score=y_pred)}
    precision_from_curve, recall_from_curve, thresholds_from_curve = precision_recall_curve(y_true, y_pred)
    prc_auc_result = {"prc-auc": auc(recall_from_curve, precision_from_curve)}

    result = {**acc_result, **precision_result, **recall_result, **f1_result, **roc_auc_result, **prc_auc_result}
    return result


model.train_model(train_df)

print("Evaluation Scores")
preds, _ = model.predict(eval_df["selfies"].tolist())
print(compute_metrics(np.ravel(eval_df["labels"].tolist()), np.ravel(preds)))

print("Test Scores")
preds, _ = model.predict(test_df["selfies"].tolist())
print(compute_metrics(np.ravel(test_df["labels"].tolist()), np.ravel(preds)))
