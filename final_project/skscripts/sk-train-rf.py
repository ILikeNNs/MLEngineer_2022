from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    valid_files = [ os.path.join(args.test, file) for file in os.listdir(args.test) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    elif len(input_files) == 1:
        train_data = pd.read_csv(input_files[0])
    
    valid_data = pd.read_csv(valid_files[0])
    
    train_y = train_data.iloc[:,0]
    train_X = train_data.iloc[:,1:]
    valid_y = valid_data.iloc[:,0]
    valid_X = valid_data.iloc[:,1:]
    
    clf = RandomForestRegressor(n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, n_jobs=-1)
    clf.fit(train_X, train_y)
    ypred = clf.predict(valid_X)
    print('rootmse:', mean_squared_error(valid_y, ypred, squared=False))
    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf