import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
# import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statistics import mean
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import common
import DTPM_utils

# Dataset selection
train_on_reduced_dataset = common.train_on_reduced_dataset

np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.2f}'.format})

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def train_model(X, y, pickle_file_name):
    if pickle_file_name == common.DTPM_regression_policy_file:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0000001, random_state=0)

    accuracy_test = 0
    model = None
    if pickle_file_name == common.DTPM_regression_policy_file:
        print("ML algorithm: Decision Tree Regressor")

        model = DecisionTreeRegressor().fit(X_train, y_train)

        predictions = model.predict(X_test)

        r_squared = metrics.r2_score(y_test, predictions)

        mae = metrics.mean_absolute_error(y_test, predictions)

        if len(y_test) > 1:
            print("R2 =", r_squared)
            print("MAE =", mae)
            print("Min", min(y), "Max", max(y), "Avg", mean(y))

    else:
        # Train a classifier
        if common.ml_algorithm == "LR":
            print("ML algorithm: Logistic regression")

            model = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=1000).fit(X_train, y_train)

            # accuracy_train = metrics.accuracy_score(y_train, lr.predict(X_train))
            accuracy_test = metrics.accuracy_score(y_test, model.predict(X_test))
        elif common.ml_algorithm == "DT":
            print("ML algorithm: Decision Tree Classifier")

            model = DecisionTreeClassifier(criterion="gini").fit(X_train, y_train)

            print("Number of nodes:", model.tree_.node_count)

            accuracy_test = metrics.accuracy_score(y_test, model.predict(X_test))

        elif common.ml_algorithm == "RF":
            print("ML algorithm: Random Forest")

            model = RandomForestClassifier(n_estimators=100, criterion="gini").fit(X_train, y_train)

            accuracy_test = metrics.accuracy_score(y_test, model.predict(X_test))

        elif common.ml_algorithm == "MLP":
            import tensorflow as tf
            from tensorflow import keras
            from keras.utils import np_utils
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Activation
            print("ML algorithm: Multi-Layer Perceptron")

            # Get model parameters
            learning_rate = args.learning_rate
            training_epochs = args.epochs
            batch_size = args.batch_size

            # Get the hidden layers topology. Each position is a hidden layer, each value is the amount of neurons in the corresponding layer.
            hidden_layers_vector = [x for x in args.hidden_layers]

            n_inputs = len(X.columns)
            print("Number of inputs:", n_inputs)
            n_outputs = 3
            number_of_hidden_layers = len(hidden_layers_vector)
            if hidden_layers_vector[0] == 0:
                number_of_hidden_layers = 0

            # Scale the data
            X_scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled_training = X_scaler.fit_transform(X_train)
            X_scaled_testing = X_scaler.transform(X_test)

            # One-hot categories
            y_train = np_utils.to_categorical(y_train, num_classes=n_outputs)
            y_test = np_utils.to_categorical(y_test, num_classes=n_outputs)

            # MLP model
            model = Sequential()
            # Hidden layer(s)
            model.add(Dense(hidden_layers_vector[0], input_dim=n_inputs, activation=tf.nn.relu))
            for hlayer in range(number_of_hidden_layers - 1):
                model.add(Dense(hidden_layers_vector[1 + hlayer], activation=tf.nn.relu))
            # Output layer
            model.add(Dense(n_outputs, activation=tf.nn.softmax))

            model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
                          loss=tf.keras.losses.categorical_crossentropy,
                          metrics=[tf.keras.metrics.categorical_accuracy])

            print(model.summary())

            model.fit(X_scaled_training, y_train, epochs=training_epochs, batch_size=batch_size, verbose=0, validation_split=0.05)

            # print("Predicting the model...")
            # predictions = model.predict(X_scaled_testing, batch_size=None, verbose=0, steps=None)

            loss, accuracy = model.evaluate(X_scaled_testing, y_test, batch_size=batch_size, verbose=0)
            accuracy_test = accuracy

        if len(y_test) > 1:
            print("Test accuracy:", accuracy_test)

    pickle.dump(model, open(pickle_file_name, 'wb'))

def main(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # start_time = time.time()

    # IL policy
    print("--- Training IL policy ---")
    if train_on_reduced_dataset:
        dataset_IL_freq_oracle = common.DATASET_FILE_DTPM.split(".")[0] + '_freq_oracle_reduced.csv'
        print("- Training on REDUCED dataset -")
    else:
        dataset_IL_freq_oracle = common.DATASET_FILE_DTPM.split(".")[0] + '_freq_oracle.csv'
        print("- Training on GLOBAL dataset -")

    print("Loading dataset... (Frequencies)")

    if not os.path.exists(dataset_IL_freq_oracle):
        print("[E] The dataset with the oracle information was not found, please run DTPM_generate_oracle.py first")
        sys.exit()

    dataset = pd.read_csv(dataset_IL_freq_oracle).drop(['Job List', 'Execution Time (s)', 'Energy Consumption (J)',
                                                        'Utilization_PE_0', 'Utilization_PE_1',
                                                        'Max_temp', 'Min_temp', 'Avg_temp', 'Throttling State'], axis=1)

    X = np.array(dataset[list(dataset)[:-1]])
    y = np.array(dataset[list(dataset)[-1]])

    train_model(X, y, common.DTPM_freq_policy_file)

    print("Loading dataset... (Num cores)")
    if train_on_reduced_dataset:
        dataset_IL_num_cores_oracle = common.DATASET_FILE_DTPM.split(".")[0] + '_num_cores_oracle_reduced.csv'
    else:
        dataset_IL_num_cores_oracle = common.DATASET_FILE_DTPM.split(".")[0] + '_num_cores_oracle.csv'

    if not os.path.exists(dataset_IL_num_cores_oracle):
        print("[E] The dataset with the oracle information was not found, please run DTPM_generate_oracle.py first")
        sys.exit()

    dataset = pd.read_csv(dataset_IL_num_cores_oracle).drop(['Job List', 'Execution Time (s)', 'Energy Consumption (J)',
                                                             'Utilization_PE_0', 'Utilization_PE_1',
                                                             'Max_temp', 'Min_temp', 'Avg_temp', 'Throttling State'], axis=1)

    X = np.array(dataset[list(dataset)[:-1]])
    y = np.array(dataset[list(dataset)[-1]])

    train_model(X, y, common.DTPM_num_cores_policy_file)

    print("Loading dataset... (Regression)")
    if train_on_reduced_dataset:
        dataset_IL_regression_oracle = common.DATASET_FILE_DTPM.split(".")[0] + '_regression_oracle_reduced.csv'
    else:
        dataset_IL_regression_oracle = common.DATASET_FILE_DTPM.split(".")[0] + '_regression_oracle.csv'

    if not os.path.exists(dataset_IL_regression_oracle):
        print("[E] The dataset with the oracle information was not found, please run DTPM_generate_oracle.py first")
        sys.exit()

    dataset = pd.read_csv(dataset_IL_regression_oracle).drop(['Job List', 'Energy Consumption (J)',
                                                             'Utilization_PE_0', 'Utilization_PE_1',
                                                             'Max_temp', 'Min_temp', 'Avg_temp', 'Throttling State'], axis=1)

    X = np.array(dataset[list(dataset)[:-1]])
    y = np.array(dataset[list(dataset)[-1]])

    train_model(X, y, common.DTPM_regression_policy_file)

    # training_time = float(float(time.time() - start_time)) / 60.0
    # print("--- {:.2f} minutes ---".format(training_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description: train and evaluate the DTPM model",
                                     epilog="""Example of use: 
                                                python3 DTPM_train_model.py -e 1000 -lr 0.005 -hl 10 8 15 -b 32""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-e", "--epochs", type=int,
                        help="define the number of epochs", default=10000)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="define the learning rate", default=0.001)
    parser.add_argument("-hl", "--hidden_layers", type=int, nargs='+',
                        help="define the hidden layers neurons, separated by spaces", default=[20])
    parser.add_argument("-s", "--single_execution", dest='single_execution', action='store_true', default=True)
    parser.add_argument("-b", "--batch_size", type=int,
                        help="define the batch size", default=256)

    args = parser.parse_args()
    main(args)
