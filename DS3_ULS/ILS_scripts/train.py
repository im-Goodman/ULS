# -*- coding: utf-8 -*-
"""
Created on Thu Sept14 13:18:59 2019

@author: skmandal"""

import os
import re
import math
import shutil
import random
import scipy.io
import numpy as np
import pandas as pd                
import matplotlib.pyplot as plt
import sklearn
import pickle

from functions import *
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor      
from sklearn.tree import DecisionTreeClassifier      
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from IPython.display import Image
from sklearn import tree
import pydotplus

from datetime import datetime
from time import sleep

import csv
import random

import itertools

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras import regularizers
from datetime import datetime
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

import sys
sys.path.append('./')

os.environ["KMP_WARNINGS"] = "FALSE" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED']=str(1000)
import random
random.seed(1000)
from numpy.random import seed
np.random.seed(1000)
import tensorflow as tf
if tf.__version__ == '2.1.0' :
    tf.random.set_seed(1000)
else :
    tf.set_random_seed(1000)

##############################################################################

app_name        = "data_IL"
classifier_types = ['RT', 'SVM', 'LiR', 'LoR', 'NN']
classifier_types = ['RT']
learning_type   = "IL"

# Setup directory
os.system('mkdir -p ./models/')

# Depths for regression tree
max_tree_depth_array = [1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,26,28,32,48,64,96,128]
max_tree_depth_array = [12]

inj_rates = [
'merged',
]

# List and iterate for different models
models = ['clustera', 'cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4']

loo_application = ''
if len(sys.argv) == 3 :
    loo_application = int(sys.argv[2])
## if len(sys.argv) > 1 :
    
if len(sys.argv) == 2 :
    dagger_string = '_dagger' + sys.argv[1]
else :
    dagger_string = ''
## if len(sys.argv) > 1 :


# Sweep classifier types
for classifier_type in classifier_types :

    # File handle to print accuracy
    accuracy_filename = './' + classifier_type + dagger_string + '_training_accuracy.rpt'
    accuracy_file = open(accuracy_filename, 'w')
    accuracy_file.close()

    # Use classification / regression
    if classifier_type == 'NN' :
        ml_type = 'classification'
    else :
        ml_type = 'classifier'
        
    # Iterate for all injection rates
    for inj_rate in inj_rates :
        
        # Iterate over each of the models
        for model in models :
            
            # Create environment with required number of labels
            if model == 'clustera' :
                num_PEs = 5
            if model == 'cluster0' :
                num_PEs = 4
            if model == 'cluster1' :
                num_PEs = 4
            if model == 'cluster2' :
                num_PEs = 2
            if model == 'cluster3' :
                num_PEs = 4
            if model == 'cluster4' :
                num_PEs = 2
    
            env = ApplicationEnv(num_PEs)
    
            # Specify the name of the file
            datafile_name = './datasets/' + app_name + '_' + model + '_' + inj_rate + dagger_string + ".csv"
            
            # Ignore a model if it is has no data
            num_lines_in_datafile = os.popen('cat ' + datafile_name).read()
            if len(num_lines_in_datafile) == 0 :
                continue
            
            # Read dataset into Pandas dataframe
            alldata_orig  = pd.read_csv("./" + datafile_name, header = 0)
            
            # Extract feature data and labels
            feature_data = alldata_orig.iloc[:, 2:31]
            
            if model == 'clustera' :
                labels = alldata_orig.iloc[:, -1]
            else :
                labels = alldata_orig.iloc[:, -2]
                if model == 'cluster1' :
                    labels = labels - 4
                if model == 'cluster2' :
                    labels = labels - 8
                if model == 'cluster3' :
                    labels = labels - 10
                if model == 'cluster4' :
                    labels = labels - 14
    
            # Check if there are more than one unique labels
            if len(np.unique(labels)) == 1 :
                print(model + " " + inj_rate + " accuracy: " + str(100.0))
                
                accuracy_file = open(accuracy_filename, 'a')
                accuracy_file.write(model + " " + inj_rate + " accuracy: " + str(100.0) + '\n')
                accuracy_file.close()
                continue
                  
            # Separate test and train data from the given dataset
            train_size = 0.5
            test_size  = 0.5
            
            feature_data_train, feature_data_test, labels_train, labels_test = \
            train_test_split(feature_data, labels, test_size=test_size, train_size=train_size, random_state=0)
            
            if loo_application != '' :
                # Exclude application from training data
                indices = np.where(feature_data_train[['JobType']] != loo_application)
            
                # Modify feature data train and labels train
                feature_data_train = feature_data_train.iloc[indices[0],:]
                labels_train = labels_train.iloc[indices[0]]

                if len(feature_data_train) == 0 :
                    print(model + " " + inj_rate + " accuracy: " + str(-1))
                    
                    accuracy_file = open(accuracy_filename, 'a')
                    accuracy_file.write(model + " " + inj_rate + " accuracy: " + str(-1) + '\n')
                    accuracy_file.close()
                    continue

            # Iterate for each tree depth of decision tree
            for max_tree_depth in max_tree_depth_array:
            
                # Specify filename to save model
                if classifier_type == "NN":
                    filename = './models/' + classifier_type + "_" + model + "_" + inj_rate + dagger_string + "_model.h5"
                else:
                    filename = './models/' + classifier_type + "_" + model + "_" + inj_rate + dagger_string + "_model_" + str(max_tree_depth) + ".sav"
    
                # Training phase
                phase = "training"
            
                # Train model with train features and labels
                ml_model = env.f_train_model(feature_data_train, labels_train, classifier_type, ml_type, max_tree_depth)
                
                # Save model
                if classifier_type == "NN":
                    ml_model.save(filename)
                else:
                    pickle.dump(ml_model, open(filename, 'wb'))
    
                # Testing phase                
                phase = "testing"
                
                # Load model
                if classifier_type == "NN":
                    regressor = load_model(filename)
                else:
                    regressor = pickle.load(open(filename, 'rb'))        
                
                # Test model and calculate accuracy
                accuracy, output_labels = env.f_test_model(classifier_type, ml_type, feature_data_test, regressor, labels_test)    
                
                print(model + " " + inj_rate + " accuracy: " + str(accuracy*100))
                
                accuracy_file = open(accuracy_filename, 'a')
                accuracy_file.write(model + " " + inj_rate + " accuracy: " + str(accuracy*100) + '\n')
                accuracy_file.close()
