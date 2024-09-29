#!/usr/bin/python
import os, re, glob, sys

dagger_iterations = 0

if len(sys.argv) > 1 :
    dagger_iterations = int(sys.argv[1])
## if len(sys.argv) > 1 :

# Define the different clusters
models = ['clustera', 'cluster0', 'cluster1', 'cluster2', 'cluster3', 'cluster4']

# Path of the training data for different injection rates
files = glob.glob('./datasets/data*')

# Extract the header information of the dataset
os.system('head -1 ' + files[0] + ' > header.rpt')

# Iterate through all the clusters
for model in models :

    # Get a list of dataset files for each cluster
    if dagger_iterations != 0 :
        files = glob.glob('./datasets/data*' + model + '*merged.csv')
        for dagger_iter in range(1, dagger_iterations + 1, 1) :
            files.extend(glob.glob('./datasets/data*' + model + '*1_dagger' + str(dagger_iter) + '*'))
    else :
        files = glob.glob('./datasets/data*' + model + '*-1.csv')
    ## if dagger_iterations != 0 :
    exec_string = 'cat '
    for file in files :
        exec_string = exec_string + ' ' + file
    exec_string = exec_string + ' > temp.csv'
    os.system(exec_string)
    os.system('grep -v Time temp.csv > temp1.csv')
    if dagger_iterations != 0 :
        os.system('cat header.rpt temp1.csv > ./datasets/data_IL_' + model + '_merged_dagger' + str(dagger_iterations) + '.csv')
    else :
        os.system('cat header.rpt temp1.csv > ./datasets/data_IL_' + model + '_merged.csv')
    ## if dagger_iterations != 0 :

os.system('rm -rf header.rpt temp.csv temp1.csv')
