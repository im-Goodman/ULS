import sys
import os, re, glob
sys.path.append('./')

import common

import DASH_Sim_v0 as ds3
import multiprocessing

## Scale values to run
scale_values_list = \
[
'20-21-1' ,
'30-31-1' ,
'40-41-1' ,
'50-51-1' ,
'65-66-1' ,
'80-81-1' ,
'100-101-1' ,
'150-151-1' ,
'175-176-1' ,
'200-201-1' ,
'250-251-1' ,
'300-301-1' ,
'400-401-1' ,
'500-501-1' ,
'600-601-1' ,
'750-751-1' ,
'1000-1001-1' ,
#'2000-2001-1' ,
#'5000-5001-1' ,
#'7500-7501-1' ,
#'10000-10001-1' ,
]

## Set dagger iteration start number
start_dagger_iteration = 1

## Set number of DAgger iterations
num_dagger_iterations = 10

## Run simulator and train through dagger iterations
for dagger_iter in range(start_dagger_iteration, start_dagger_iteration + num_dagger_iterations, 1) :

    ## Set dagger iteration variable in common.py
    common.ils_dagger_iter = dagger_iter

    ## Run simulator
    p = multiprocessing.Pool(50)
    p.map(ds3.run_simulator, scale_values_list)

    ## Merge training datasets
    os.system('python ILS_scripts/merge_dataset.py ' + str(dagger_iter))

    ## Train policies
    os.system('python ILS_scripts/train.py ' + str(dagger_iter))

    ## Run simulation if it is the final iteration
    if dagger_iter == start_dagger_iteration + num_dagger_iterations -1 :
         common.ils_dagger_iter = dagger_iter + 1

         ## Run simulator
         p = multiprocessing.Pool(50)
         p.map(ds3.run_simulator, scale_values_list)

