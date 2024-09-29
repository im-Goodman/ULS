'''
Description: This file contains the script to generate the traces of several configurations at once.
'''

import os
import sys
import itertools
import multiprocessing
import fnmatch
import pandas as pd
import time
import numpy, itertools
import csv
import shutil

import common
import DASH_Sim_v0
import DASH_Sim_utils
import configparser
import DASH_SoC_parser
import generate_traces

# Configurations are defined on generate_traces.py

resource_matrix = common.ResourceManager()  # This line generates an empty resource matrix

if __name__ == '__main__':
    start_time = time.time()
    DASH_Sim_utils.clean_traces()
    # Parse the resource file
    DVFS_config = configparser.ConfigParser()
    DVFS_config.read('config_file.ini')
    resource_file = DVFS_config['DEFAULT']['resource_file']
    # Update the number os PEs in the common.py file
    DASH_SoC_parser.resource_parse(resource_matrix, resource_file)  # Parse the input configuration file to populate the resource matrix

    if generate_traces.heterogeneous_PEs:
        DVFS_config_list_prod = itertools.product(*generate_traces.DVFS_modes)
    else:
        DVFS_config_list_prod = itertools.product(generate_traces.DVFS_modes, repeat=common.num_PEs_TRACE)

    common.CLEAN_TRACES = False
    common.generate_complete_trace = True
    common.TRACE_FREQUENCY = False
    common.TRACE_PES = False
    common.enable_real_time_constraints = False
    common.enable_num_cores_prediction = False
    common.TRACE_TEMPERATURE = True
    common.TRACE_SYSTEM = True

    for DVFS_config in DVFS_config_list_prod:
        if len(DVFS_config) < 2:
            print("[E] Trace generation must have at least little and big clusters, check generate_traces_workload")
            sys.exit()
        freq_little = float(DVFS_config[0].split('-')[1]) / 1000
        freq_big = float(DVFS_config[1].split('-')[1]) / 1000
        for N_little in generate_traces.N_little_list:
            for N_big in generate_traces.N_big_list:
                common.DVFS_cfg_list = DVFS_config
                common.gen_trace_capacity_little = N_little
                common.gen_trace_capacity_big = N_big
                DASH_Sim_v0.run_simulator()
                csv_file = pd.read_csv(common.TRACE_FILE_TEMPERATURE)
                avg_temperature = float(csv_file['Temperature'].mean())
                row = [common.job_list, DVFS_config, N_little, N_big]
                row.append(avg_temperature)
                if not os.path.exists(common.TRACE_FILE_TEMPERATURE_WORKLOAD):
                    with open(common.TRACE_FILE_TEMPERATURE_WORKLOAD, 'w', newline='') as csvfile:
                        result_file = csv.writer(csvfile, delimiter=',')
                        result_file.writerow(["Job_list", "DVFS_config", "N_little", "N_big", "Temperature"])
                with open(common.TRACE_FILE_TEMPERATURE_WORKLOAD, 'a', newline='') as csvfile:
                    result_file = csv.writer(csvfile, delimiter=',')
                    result_file.writerow(row)
                os.remove(common.TRACE_FILE_TEMPERATURE)

    shutil.copy("trace_system__0.csv", "trace_system_workload.csv")

    sim_time = float(float(time.time() - start_time)) / 60.0
    print("--- {:.2f} minutes ---".format(sim_time))
