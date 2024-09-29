'''
Description: This file contains the script to generate the traces of several DVFS configurations at once.
'''

import os
import sys
import itertools
import multiprocessing
import fnmatch
import pandas as pd
import csv
import common
import DASH_Sim_v0
import DASH_Sim_utils
import configparser
import DASH_SoC_parser
import DTPM_utils
import processing_element
import time
import random
import ast
import generate_traces

# Define the thresholds for the deadline generation (0 - 1)
deadline_low_threshold  = 0.05
deadline_high_threshold = 0.10

# Define the number of snippets
num_snippets = 50

resource_matrix = common.ResourceManager()  # This line generates an empty resource matrix

def generate_bursts(burst_size, gap_size):
    job_list_sample = []
    burst_index = 0
    gap_index = len(sorted_dataset.index) - 1
    dataset_size = len(sorted_dataset.index)
    insert_burst = True
    i = 0
    high_temp_sample = list(sorted_dataset['Job List'][0 : int(0.05 * dataset_size)])
    low_temp_sample  = list(sorted_dataset['Job List'][int(0.95 * dataset_size) : dataset_size - 1])
    while i < num_snippets:
        if insert_burst:
            for burst in range(burst_size):
                idx = random.randint(0, len(high_temp_sample) - 1)
                snippet = high_temp_sample.pop(idx)
                job_list_sample.append(ast.literal_eval(snippet))
                burst_index += 1
            i += burst_size
            insert_burst = False
        else:
            for gap in range(gap_size):
                idx = random.randint(0, len(low_temp_sample) - 1)
                snippet = low_temp_sample.pop(idx)
                job_list_sample.append(ast.literal_eval(snippet))
                gap_index -= 1
            i += gap_size
            insert_burst = True
    print("#job_list =", job_list_sample)

if __name__ == '__main__':
    random.seed(1)
    start_time = time.time()
    # Parse the resource file
    config = configparser.ConfigParser()
    config.read('config_file.ini')
    resource_file = config['DEFAULT']['resource_file']
    # Update the number os PEs in the common.py file
    DASH_SoC_parser.resource_parse(resource_matrix, resource_file)  # Parse the input configuration file to populate the resource matrix

    dataset = pd.read_csv(common.DATASET_FILE_DTPM.split('.')[0] + " - Initial.csv")

    # Generate deadlines
    grouped_snippet = dataset.sort_index().groupby(['Job List'])
    print("Num_snippets:", len(grouped_snippet))

    with open(common.DEADLINE_FILE, 'w', newline='') as csvfile:
        deadline_writer = csv.writer(csvfile, delimiter=',')

        header = ["Job List", "Deadline"]
        deadline_writer.writerow(header)

        for index, snippet in grouped_snippet:
            min_exec_time = min(snippet['Execution Time (s)'])
            max_exec_time = max(snippet['Execution Time (s)'])

            deadline_t = random.uniform(deadline_low_threshold, deadline_high_threshold)
            deadline = (max_exec_time - min_exec_time) * deadline_t + min_exec_time

            print("{}, {:.8f}, Ratio {:.2f}, Min {:.8f}, Max {:.8f}".format(index, deadline, deadline_t, min_exec_time, max_exec_time))

            deadline_writer.writerow([index, deadline])

    # Generate job_list
    job_list_complete = DTPM_utils.multinomial_combinations(generate_traces.N_jobs, generate_traces.N_applications)
    print("Job_list length:", len(job_list_complete))
    print("# --- Random sample ---")
    sample = random.sample(job_list_complete, num_snippets)
    job_list_sample = []
    for s in sample:
        job_list_sample.append(list(s))
    print("#job_list =", job_list_sample)

    # Parse dataset file to identify high temp microbenchmarks
    pd.set_option('mode.chained_assignment', None)
    dataset_filtered = dataset[(dataset['N_little'] == 4) & (dataset['N_big'] == 4) &
                               (dataset['FREQ_PE_0 (GHz)'] == 1.4) & (dataset['FREQ_PE_1 (GHz)'] == 2.0)]
    sorted_dataset = dataset_filtered.sort_values(by=['Max_temp'], ascending=False).reset_index()
    # print(sorted_dataset[['Job List', 'Max_temp']].head(num_snippets))

    print("# ---- Burst 3x10 -----")
    burst_size = 10
    gap_size   = 10
    generate_bursts(burst_size, gap_size)

    print("# ---- Burst 2x20 -----")
    burst_size = 20
    gap_size   = 10
    generate_bursts(burst_size, gap_size)

    sim_time = float(float(time.time() - start_time)) / 60.0
    print("--- {:.2f} minutes ---".format(sim_time))
