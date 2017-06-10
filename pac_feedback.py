#!/usr/bin/env python3

import pandas as pd

from functools import reduce

from sklearn.grid_search import ParameterGrid

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.externals import joblib

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

# ## Configuration

# Data
DATA_DIR = "data"
TRAIN_NUMERIC = "%s/%s" % (DATA_DIR, "train_numeric.csv")
TEST_NUMERIC = "%s/%s" % (DATA_DIR, "test_numeric.csv")

# Cache
CACHE_DIR = "cache"
MODEL_CACHE = "%s/%s" % (CACHE_DIR, "passive_aggressive_classifier.pkl")
KFOLD_OFFSETS_CACHE = "%s/%s" % (CACHE_DIR, "kfold_offsets.pkl")
BEST_PARAMS_CACHE = "%s/%s" % (CACHE_DIR, "best_params.pkl")

# Submission
SUBMISSION_DIR = "submission"
SUBMISSION_FILE = "%s/%s" % (SUBMISSION_DIR, "pac_feedback.csv.gz")

# How many records load in memory per chunk
CHUNK_SIZE = 50000

# Random seed
SEED = 23821

# Weights associated to classes
CLASS_WEIGHTS = {1: 0.9, 0: 0.1}

# K-fold cross validation: how many folds
CV_N_FOLDS = 3

# Verbosity level
VERBOSE = 1

# ID column
ID_COL = "Id"

# Label column
LABEL_COL = "Response"


def feature_engineering(chunk):

    def get_lines_passed(x):

        lines_passed = x.dropna().index.map(lambda x: x[0:2])

        lines = [0, 0, 0, 0]

        if "L0" in lines_passed:
            lines[0] = 1
        if "L1" in lines_passed:
            lines[1] = 1
        if "L2" in lines_passed:
            lines[2] = 1
        if "L3" in lines_passed:
            lines[3] = 1

        return "".join(map(str, lines))

    chunk["lines_tmp"] = chunk.apply(get_lines_passed, axis=1)

    chunk["is_line_0"] = chunk["lines_tmp"].map(lambda x: 1 if x[0] == "1" else 0)
    chunk["is_line_1"] = chunk["lines_tmp"].map(lambda x: 1 if x[1] == "1" else 0)
    chunk["is_line_2"] = chunk["lines_tmp"].map(lambda x: 1 if x[2] == "1" else 0)
    chunk["is_line_3"] = chunk["lines_tmp"].map(lambda x: 1 if x[3] == "1" else 0)

    chunk.drop(["lines_tmp"], axis=1, inplace=True)

    chunk = chunk.fillna(-1)
    return chunk


def train(datafile, model, params, chunk_size):

    df_iter = pd.read_csv(
        datafile, header=0, sep=",",
        encoding="utf-8", chunksize=chunk_size)

    model = model()
    model.set_params(**params)

    for (index, chunk) in enumerate(df_iter):
        print(" - elaborating chunk %d" % index)

        chunk = feature_engineering(chunk)

        model.partial_fit(
            chunk.drop([ID_COL, LABEL_COL], axis=1).values,
            chunk[LABEL_COL].values, [0, 1])

        y_pred = list(model.predict(chunk.drop([ID_COL, LABEL_COL], axis=1).values))
        y_true = chunk[LABEL_COL]

        mcc = matthews_corrcoef(y_true, y_pred)
        print("   + MCC %0.6f" % mcc)

    return model


def load_or_train(datafile, model, params, chunk_size, dump):
    try:
        model = joblib.load(dump)
        print("Loading model from cache (%s)" % dump)
    except FileNotFoundError:
        print("Fitting model with train dataset")
        model = train(datafile, model, params, chunk_size)
        print("Dumping model in %s" % dump)
        joblib.dump(model, dump)
    finally:
        return model


def predict(datafile, model, chunk_size):
    y_pred = []
    y_true = None
    ids = []

    df_iter = pd.read_csv(
        datafile, header=0, sep=",",
        encoding="utf-8", chunksize=chunk_size)

    for chunk in df_iter:

        chunk = feature_engineering(chunk)

        cols_to_remove = [ID_COL]
        if LABEL_COL in chunk.columns:
            cols_to_remove.append(LABEL_COL)
            if y_true is None:
                y_true = []
            y_true += list(chunk[LABEL_COL])

        ids += list(chunk[ID_COL])

        y_pred += list(model.predict(chunk.drop(cols_to_remove, axis=1).values))

    return ids, y_pred, y_true


def kfold(datafile, n_folds, chunk_size):
    """Computes chunks indices for each validation fold.

    Given a datafile, the desired number of folds, and the chunk size
    chosen for batch processing, returns a dictionary indicating, for
    each fold, the indices of the chunks belonging to that fold.

    Args:
        datafile: a path of a CSV data file
        n_folds: the desired number of folds
        chunk_size: the size of the chunk for batch processing

    Returns:
        A dict containing folds to indices associations.
        Example:

        {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8], 2: [9, 10, 11]}
    """

    n_lines = 0
    n_chunks = 0

    df_iter = pd.read_csv(
        datafile, header=0, sep=",",
        encoding="utf-8", chunksize=chunk_size)

    for chunk in df_iter:
        n_lines += len(chunk)
        n_chunks += 1

    chunks_per_fold = int(n_chunks / n_folds)

    kfold_offsets = dict()

    fold = 0
    for n in range(0, n_chunks):

        if fold not in kfold_offsets:
            kfold_offsets[fold] = list()

        kfold_offsets[fold].append(n)

        if n != 0 and n % chunks_per_fold == 0 and fold + 1 <= n_folds - 1:
            fold += 1

    return kfold_offsets


def load_or_compute_kfold(datafile, n_folds, chunk_size, dump):
    """Retrieves K-fold split from cache or compute it, if needed.

    Args:
        datafile: a path of a CSV data file
        n_folds: the desired number of folds
        chunk_size: the size of the chunk for batch processing
        dump: the cache file path

    Returns:
        A dict containing folds to indices associations.
        Example:

        {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8], 2: [9, 10, 11]}
    """
    try:
        kfold_offsets = joblib.load(dump)
        print("Loading kfold offsets from cache (%s)" % dump)
    except FileNotFoundError:
        print("Estimating offsets for K-Fold cross-validation")
        print(" - number of folds: %d" % n_folds)
        kfold_offsets = kfold(datafile, n_folds, chunk_size)
        print(" - dumping offsets in %s" % dump)
        joblib.dump(kfold_offsets, dump)
    finally:
        print(" - offsets: %s" % kfold_offsets)
        return kfold_offsets


def cross_validate(datafile, model, kfold_offsets, chunk_size):
    scores = []

    for (fold, offsets) in kfold_offsets.items():

        predictions = []
        responses = []

        df_iter = pd.read_csv(
            datafile, header=0, sep=",",
            encoding="utf-8", chunksize=chunk_size)

        for (index, chunk) in enumerate(df_iter):

            if index not in offsets:

                if VERBOSE:
                    print("Fitting with fold %d" % index)

                chunk = feature_engineering(chunk)

                model.partial_fit(
                    chunk.drop([ID_COL, LABEL_COL], axis=1).values,
                    chunk[LABEL_COL].values, [0, 1])

        df_iter = pd.read_csv(
            datafile, header=0, sep=",",
            encoding="utf-8", chunksize=chunk_size)

        for (index, chunk) in enumerate(df_iter):

            if index in offsets:

                if VERBOSE:
                    print("Predicting with fold %d" % index)

                chunk = feature_engineering(chunk)

                predictions += list(model.predict(
                    chunk.drop([ID_COL, LABEL_COL], axis=1).values))
                responses += list(chunk[LABEL_COL].values)

        mcc = matthews_corrcoef(responses, predictions)
        scores.append(mcc)
        print("Matthews corrcoef: %0.6f" % mcc)
        print("Confusion matrix:")
        print(confusion_matrix(responses, predictions))

    return scores


def grid_search_cv(datafile, model, grid_params, chunk_size):
    print("Starting batch with chunk size of %d lines" % chunk_size)

    kfold_offsets = load_or_compute_kfold(
        datafile, CV_N_FOLDS, chunk_size, KFOLD_OFFSETS_CACHE)

    best_mcc = None
    best_params = None

    for param in list(ParameterGrid(grid_params)):

        model_obj = model()

        model_obj.set_params(**param)

        if best_params is None:
            best_params = model_obj.get_params()

        print("Evaluating model %s" % model_obj.get_params())

        scores = cross_validate(datafile, model_obj, kfold_offsets, chunk_size)
        avg_score = reduce(lambda x, y: x + y, scores) / len(scores)
        print("Average MCC %0.6f" % avg_score)

        if best_mcc is None:
            best_mcc = avg_score
        else:
            if avg_score > best_mcc:
                best_mcc = avg_score
                best_params = model_obj.get_params()

    return best_mcc, best_params


def load_or_grid_search_cv(datafile, model, grid_params, chunk_size, dump):
    try:
        best_mcc, best_params = joblib.load(dump)
        print("Loading best params from cache (%s)" % dump)
    except FileNotFoundError:
        print("Performing grid search with cross-validation")
        (best_mcc, best_params) = grid_search_cv(
            datafile, model, grid_params, chunk_size)
        print(" - dumping best params in %s" % dump)
        joblib.dump((best_mcc, best_params), dump)
    finally:
        return best_mcc, best_params


def save_submission(ids, y_true, file):
    df_csv = pd.DataFrame()
    df_csv[ID_COL] = pd.Series(ids)
    df_csv[LABEL_COL] = pd.Series(y_true)
    df_csv.to_csv(file, columns=[ID_COL, LABEL_COL], index=False, encoding="utf-8", compression="gzip")


def main():

    grid_params = {
        "class_weight": [{1: 0.95, 0: 0.05}],
        "random_state": [SEED],
        "warm_start": [True],
        "loss": ["squared_hinge"],
        "C": [0.7]
    }

    (best_mcc, best_params) = load_or_grid_search_cv(
        TRAIN_NUMERIC, PassiveAggressiveClassifier, grid_params,
        CHUNK_SIZE, BEST_PARAMS_CACHE)

    print("Best MCC: %0.6f" % best_mcc)
    print("Best model params: %s" % best_params)

    model = load_or_train(TRAIN_NUMERIC, PassiveAggressiveClassifier, best_params, CHUNK_SIZE, MODEL_CACHE)

    print("Elaborating predictions on train dataset")
    (ids, y_pred, y_true) = predict(TRAIN_NUMERIC, model, CHUNK_SIZE)

    mcc = matthews_corrcoef(y_true, y_pred)
    print("In-sample MCC %0.6f" % mcc)
    print("In-sample confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("Elaborating predictions on test dataset")
    (ids, y_pred, y_true) = predict(TEST_NUMERIC, model, CHUNK_SIZE)

    print("Got predictions: %d entries" % (len(y_pred)))

    save_submission(ids, y_pred, SUBMISSION_FILE)


if __name__ == "__main__":
    main()
