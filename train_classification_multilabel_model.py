import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from simpletransformers.classification import MultiLabelClassificationModel

import pandas as pd
import numpy as np
import chemprop

from pandarallel import pandarallel
import to_selfies

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


num_labels = len(pd.read_csv(args.dataset).columns)-1
model_args = {
    "num_train_epochs": 25,
    "learning_rate": 1e-5,
    "weight_decay": 0.1,

    "output_dir": args.save_to,
}
model = MultiLabelClassificationModel("roberta", args.model, num_labels=num_labels, use_cuda=True, args=model_args)

from prepare_finetuning_data import train_val_test_split_multilabel
(train, val, test) = train_val_test_split_multilabel(args.dataset)

train_smiles = [item[0] for item in train.smiles()]
validation_smiles = [item[0] for item in val.smiles()]
test_smiles = [item[0] for item in test.smiles()]

train_df = pd.DataFrame(np.column_stack([train_smiles, train.targets()]), columns = ['smiles'] + ["Feature_" + str(i) for i in range(len(train.targets()[0]))])
eval_df = pd.DataFrame(np.column_stack([validation_smiles, val.targets()]), columns = ['smiles'] + ["Feature_" + str(i) for i in range(len(val.targets()[0]))])
test_df = pd.DataFrame(np.column_stack([test_smiles, test.targets()]), columns = ['smiles'] + ["Feature_" + str(i) for i in range(len(test.targets()[0]))])

def smiles_to_selfies(df):
    df.insert(0, "selfies", df["smiles"])
    pandarallel.initialize()
    df.selfies = df.selfies.parallel_apply(to_selfies.to_selfies)

    df.drop(df[df.smiles == df.selfies].index, inplace=True)
    df.drop(columns=["smiles"], inplace=True)

    return df

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
acc = load_metric('accuracy')
precision = load_metric('precision')
recall = load_metric('recall')
f1 = load_metric('f1')

def compute_metrics(y_true, y_pred):
    acc_result = acc.compute(predictions=y_pred, references=y_true)
    precision_result = precision.compute(predictions=y_pred, references=y_true)
    recall_result = recall.compute(predictions=y_pred, references=y_true)
    f1_result = f1.compute(predictions=y_pred, references=y_true)
    roc_auc_result = {"roc-auc": roc_auc_score(y_true = y_true, y_score= y_pred)}
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