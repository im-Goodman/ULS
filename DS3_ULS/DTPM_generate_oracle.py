import os
import sys
import pandas as pd
import time
import re
import configparser

import DTPM_utils
import DASH_SoC_parser
import common

resource_matrix = common.ResourceManager()  # This line generates an empty resource matrix
config = configparser.ConfigParser()
config.read('config_file.ini')
resource_file = config['DEFAULT']['resource_file']
DASH_SoC_parser.resource_parse(resource_matrix, resource_file)                  # Parse the input configuration file to populate the resource matrix
start_time = time.time()
common.oracle_config_dict = DTPM_utils.get_oracle_frequencies_and_num_cores()

def generate_oracle():
    if os.path.exists(common.DATASET_FILE_DTPM):
        print("Loading dataset and obtaining the oracle frequencies...")
        sim_time = float(float(time.time() - start_time)) / 60.0
        print("{:.2f} minutes".format(sim_time))
        DTPM_utils.add_oracle_to_dataset('Frequency')
        DTPM_utils.create_reduced_dataset('Frequency')
        DTPM_utils.add_oracle_to_dataset('Num_cores')
        DTPM_utils.create_reduced_dataset('Num_cores')
        DTPM_utils.add_oracle_to_dataset('Regression')
        DTPM_utils.create_reduced_dataset('Regression')
        sim_time = float(float(time.time() - start_time)) / 60.0
        print("--- {:.2f} minutes ---".format(sim_time))
        print("DONE: dataset is saved...")
    else:
        print("[E] Dataset file not found:", common.DATASET_FILE_DTPM)
        sys.exit()

if __name__ == '__main__':
    generate_oracle()