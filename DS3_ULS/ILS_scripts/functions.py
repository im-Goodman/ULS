# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:19:36 2019

@author: skmandal, anallam1
"""

import os
import math
import shutil
import random
import scipy.io
import numpy as np
import pandas as pd                
import matplotlib.pyplot as plt

from functions import *
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression   
from sklearn.linear_model import LogisticRegression   
from sklearn.svm import SVC
from datetime import datetime
from time import sleep

import csv
import random

#for plotting
from IPython.display import clear_output 
# for Earlystop

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import optimizers
from keras import regularizers
from datetime import datetime
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.optimizers import Adadelta
from keras.optimizers import rmsprop
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.utils.np_utils import to_categorical


class ApplicationEnv:
    def __init__(self, num_PEs):
        self.num_PEs = num_PEs
        
    def f_train_model(self, feature_data,  labels, classifier_type, ml_type, max_tree_depth):
        
        X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.20, train_size=0.80, random_state=0)
        
        if classifier_type == 'RT':
            #regressor = DecisionTreeRegressor(max_depth = max_tree_depth)
            regressor = DecisionTreeClassifier(max_depth = max_tree_depth)
            regressor.fit(X_train, y_train)
        elif classifier_type == 'LiR':
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
        elif classifier_type == 'LoR':
            regressor = LogisticRegression()
            regressor.fit(X_train, y_train)
        elif classifier_type == 'SVM' :
            regressor = SVC()
            regressor.fit(X_train, y_train)
        elif classifier_type == 'NN':
            regressor = self.f_build_model(feature_data.shape[1], ml_type)

            if ml_type == 'classification' :
                to_model_y_train = to_categorical(y_train, num_classes=self.num_PEs)
                to_model_y_test  = to_categorical(y_test, num_classes=self.num_PEs)
            else :
                to_model_y_train = y_train
                to_model_y_test  = y_test
            
            regressor.fit(X_train, to_model_y_train, validation_data=(X_test, to_model_y_test), batch_size=256, 
                          epochs=100, verbose=0, shuffle=True) 
        else:
            raise Exception('Unexpected classifier type')
            
        return regressor
       
    def f_test_model(self, classifier_type, ml_type, feature_data, regressor, labels):
        
        data_length = len(feature_data)
        output_labels = -1 * np.ones(len(feature_data))
        num_correct_pred = 0
        
        for data_idx in range(0, data_length):
            
            data = feature_data.iloc[[data_idx]]
            if classifier_type == 'NN' :
                if ml_type == 'classification' :
                    output_label = np.argmax(regressor.predict(data))
                else :
                    output_label = np.around(regressor.predict(data))
                output_labels[data_idx] = output_label
                if output_label == labels.iloc[data_idx] :
                    num_correct_pred += 1
            else :
                if ml_type == 'classification' :
                    output_label = int(regressor.predict(data))
                else :
                    output_label = np.around(regressor.predict(data))
                output_labels[data_idx] = output_label
                if output_label == labels.iloc[data_idx] :
                    num_correct_pred += 1
            
        return (num_correct_pred / data_length), output_labels
       
    def f_build_model(self, num_features, ml_type):
        # Neural Net Model
        model = Sequential()
        
        model.add(Dense(32, input_dim=num_features, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        
        if ml_type == 'classification' :
            model.add(Dense(self.num_PEs, activation='softmax'))
        else :
            model.add(Dense(1, activation='relu'))
            
        # Compile the model. Loss function is categorical crossentropy
        if ml_type == 'classification' :
            model.compile(loss='binary_crossentropy', 
                          optimizer=Adam(lr=0.001), 
                          metrics=['accuracy'])    
        else :
            model.compile(loss='mse', 
                          optimizer=Adam(lr=0.001), 
                          metrics=['mean_absolute_error'])        
        
        return model
