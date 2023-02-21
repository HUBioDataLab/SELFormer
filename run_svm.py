import os
import sys
from time import time
from fnmatch import fnmatch
import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

from prepare_svm_data import *


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, metavar="/path/to/dataset/", help="Path of the input MoleculeNet datasets.")
parser.add_argument("--model_file", required=True, metavar="<str>", type=str, help="Name of the pretrained model.")
parser.add_argument("--generate_embeddings", required=False, metavar="<int>", type=int, default=0, help="Generate embeddings for MoleculeNet datasets using the pre-trained model (1) or use pre-generated embeddings (0). To download the pre-generated embeddings, follow the instructions in the README.md file. Default: 0")
parser.add_argument("--run_svm", required=False, metavar="<int>", type=int, default=1, help="Run SVM on the pre-trained model (1) or not (0). Default: 1")

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_file = args.model_file # path of the pre-trained model
config = RobertaConfig.from_pretrained(model_file)
config.output_hidden_states = True
tokenizer = RobertaTokenizer.from_pretrained("./data/RobertaFastTokenizer")
model = RobertaModel.from_pretrained(model_file, config=config)


def compute_metrics(predictions, labels, task):
    """
    Computes the metrics for a given task.
    :param predictions: model predictions. For classification tasks, it is a tuple of (preds, prob_predictions). 
    :param labels: ground truth labels
    :param task: task name
    :return: dict of metrics
    """

    if 'classification' in task:
        preds = predictions[0]
        prob_predictions = predictions[1]

        auroc = metrics.roc_auc_score(labels, prob_predictions),
        acc_result =  metrics.accuracy_score(labels, preds),
        precision_result = metrics.precision_score(labels, preds),
        recall_result = metrics.recall_score(labels, preds),
        f1_result = metrics.f1_score(labels, preds),
        precision_from_curve, recall_from_curve, thresholds_from_curve = metrics.precision_recall_curve(labels, prob_predictions)
        prc_auc_result = metrics.auc(recall_from_curve, precision_from_curve)

        # return results dict
        result = {
            "accuracy": round(acc_result[0], 3),
            "precision": round(precision_result[0], 3),
            "recall": round(recall_result[0], 3),
            "f1": round(f1_result[0], 3),
            "roc-auc": round(auroc[0], 3),
            "prc-auc": prc_auc_result
        }
        
    elif task == 'regression':
        mse = metrics.mean_squared_error(labels, predictions, squared=True)
        rmse = metrics.mean_squared_error(labels, predictions, squared=False)
        mae = metrics.mean_absolute_error(labels, predictions)

        result = { 
            "mse": round(mse, 3),
            "rmse": round(rmse, 3),
            "mae": round(mae, 3)
        }

    return result


def run_classification(root, model_name, task):
    """
    Runs classification task on the pre-trained model.
    :param root: path of the pre-generated embeddings
    :param pattern: pattern of the pre-generated embeddings
    :param model_name: name of the pre-trained model
    :param task: task name (binary or multilabel)
    :return: results_df
    """

    results_list = []
    pattern = f"*{model_name}_embeddings.csv"

    if task == 'binary':
        for path, subdirs, files in os.walk(root):
            for name in files:

                # classification
                for dataset in BINARY_CLASSIFICATION:

                    if fnmatch(name, pattern) and dataset == name.split('_model')[0]:
                        print(f"Running {dataset} classification with {model_name} embeddings")
                        t0 = time()
                        train_df, test_df = train_val_test_split(os.path.join(path, name), TARGET_COLUMNS[dataset])
                        
                        train_features = np.vstack(train_df['sequence_embeddings'])
                        train_labels = train_df['target']
                        test_features = np.vstack(test_df['sequence_embeddings'])
                        test_labels = test_df['target']

                        model = SVC(C=5.0, probability=True)
                        model.fit(train_features, train_labels)
                        preds = model.predict(test_features)
                        prob_preds = model.predict_proba(test_features)[:, 1]
                        result = compute_metrics(predictions=[preds, prob_preds], labels=test_labels, task='binary_classification')
                        
                        # save results to results_df
                        results_list.append([dataset, model_name, result['accuracy'], result['precision'], result['recall'], result['f1'], result['roc-auc'], result['prc-auc']])
                        t1 = time()
                        print(f'Finished in {round((t1-t0) / 60, 2)} mins')

    elif task == 'multilabel':
        for path, subdirs, files in os.walk(root):
            for name in files:

                # classification
                for dataset in MULTILABEL_CLASSIFICATION:

                    if fnmatch(name, pattern) and dataset == name.split('_model')[0]:
                        print(f"Running {dataset} multi-label classification with {model_name} embeddings")
                        train_df, test_df = train_val_test_split_multilabel(os.path.join(path, name))
                        
                        train_features = np.vstack(train_df['sequence_embeddings'])
                        train_labels = train_df.drop(columns=['sequence_embeddings'])
                        test_features = np.vstack(test_df['sequence_embeddings'])
                        test_labels = test_df.drop(columns=['sequence_embeddings'])

                        model = MultiOutputClassifier(SVC(C=5.0, probability=True), n_jobs=-1)
                        model.fit(train_features, train_labels)
                        preds = model.predict(test_features)
                        preds = np.ravel(preds)

                        prob_predictions = model.predict_proba(test_features)
                        y_prob_pred = np.transpose([pred[:, 1] for pred in prob_predictions])
                        y_prob_pred = np.ravel(y_prob_pred)

                        test_labels = np.ravel(test_labels.values)
                        result = compute_metrics(predictions=[preds, y_prob_pred], labels=test_labels, task='multilabel_classification')
                        
                        # save results to results_df
                        results_list.append([dataset, model_name, result['accuracy'], result['precision'], result['recall'], result['f1'], result['roc-auc'], result['prc-auc']])

    columns = ['dataset', 'model', 'accuracy', 'precision', 'recall', 'f1', 'roc-auc', 'prc-auc']
    results_df = pd.DataFrame(results_list, columns=columns)

    return results_df


def run_regression(root, model_name):

    results_list = []
    pattern = f"*{model_name}_embeddings.csv"

    for path, subdirs, files in os.walk(root):
        for name in files:

            # classification
            for dataset in REGRESSION:

                if fnmatch(name, pattern) and dataset == name.split('_model')[0]:
                    print(f"\nRunning {dataset} regression with {model_name} embeddings")
                    train_df, test_df = train_val_test_split(os.path.join(path, name), TARGET_COLUMNS[dataset])
                    
                    train_features = np.vstack(train_df['sequence_embeddings'])
                    train_labels = train_df['target']
                    test_features = np.vstack(test_df['sequence_embeddings'])
                    test_labels = test_df['target']

                    model = SVR(C=5.0)
                    model.fit(train_features, train_labels)
                    preds = model.predict(test_features)
                    result = compute_metrics(predictions=preds, labels=test_labels, task='regression')
                    
                    # save results to results_df
                    results_list.append([dataset, model_name, result['mse'], result['rmse'], result['mae']])

    columns = ['dataset', 'model', 'mse', 'rmse', 'mae']
    results_df = pd.DataFrame(results_list, columns=columns)

    return results_df


def generate_embeddings(model_file, args):

    root = args.dataset_path
    model_name = model_file.split("/")[-1]

    prepare_data_pattern = "*.csv"

    print(f"\nGenerating embeddings using pre-trained model {model_name}")
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, prepare_data_pattern) and not any(substring in name for substring in ['selfies', 'embeddings', 'results']):
                dataset_file = os.path.join(path, name)
                generate_moleculenet_selfies(dataset_file)

                selfies_file = os.path.join(path, name.split(".")[0] + "_selfies.csv")
                generate_moleculenet_embeddings(selfies_file, model_name, tokenizer, model)


def run_svm(model_file, args):

    root = args.dataset_path
    model_name = model_file.split("/")[-1]

    print("\nTraining SVM models")

    binary_classification_results = run_classification(root, model_name, 'binary')
    binary_classification_result_file = os.path.join(root, f"{model_name}_binary_classification_results.csv")
    binary_classification_results.to_csv(binary_classification_result_file, index=False)
    print(f"\nSaved binary classification results to {binary_classification_result_file}")

    multilabel_classification_results = run_classification(root, model_name, 'multilabel')
    multilabel_classification_result_file = os.path.join(root, f"{model_name}_multilabel_classification_results.csv")
    multilabel_classification_results.to_csv(multilabel_classification_result_file, index=False)
    print(f"\nSaved multilabel classification results to {multilabel_classification_result_file}")

    regression_results = run_regression(root, model_name)
    regression_result_file = os.path.join(root, f"{model_name}_regression_results.csv")
    regression_results.to_csv(regression_result_file, index=False)
    print(f"\nSaved regression results to {regression_result_file}")

BINARY_CLASSIFICATION = ['bace', 'bbbp', 'hiv', 'tox21']
REGRESSION = ['esol', 'freesolv', 'lipo', 'pdbbind_full']
MULTILABEL_CLASSIFICATION = ['sider', 'tox21']

TARGET_COLUMNS = {
    'bace': 'Class',
    'bbbp': 'p_np',
    'hiv': 'HIV_active',
    'esol': 'measured log solubility in mols per litre',
    'freesolv': 'freesolv',
    'lipo': 'lipo',
    'pdbbind_full': '-logKd/Ki',
    'tox21': 'SR-p53',
}

if args.generate_embeddings == 1:
    # generate embeddings for each dataset
    generate_embeddings(model_file, args)

if args.run_svm == 1:
    if args.generate_embeddings == 0:
        model_name = model_file.split("/")[-1]
        # check if there are embeddings for each dataset
        for dataset in BINARY_CLASSIFICATION + REGRESSION + MULTILABEL_CLASSIFICATION:
            if dataset in BINARY_CLASSIFICATION or dataset in MULTILABEL_CLASSIFICATION:
                task = 'classification'
            else:
                task = 'regression'
            if not os.path.exists(os.path.join(args.dataset_path, task, dataset, f"{dataset}_{model_name}_embeddings.csv")):
                print(f"\nEmbedding file for {dataset} is not found. Download embeddings by following the instructions in the README.md file or generate embeddings by running the script with the --generate_embeddings=1 flag")
                sys.exit()
        
    run_svm(model_file, args)




