"""
Description: This file contains all the common parameters used in DASH_Sim.
"""
import platform
# from IPython import  get_ipython
import configparser
from numpy import random
import sys
import os
import ast

import networkx as nx
import pickle
import numpy as np

time_at_sim_termination = -1

output = {}

##### Added for ILS
ils_clustera_filename_header = 0
ils_clustera_filename = ''
ils_clustera_fp = ''
ils_cluster0_filename = ''
ils_cluster0_fp = ''
ils_cluster1_filename = ''
ils_cluster1_fp = ''
ils_cluster2_filename = ''
ils_cluster2_fp = ''
ils_cluster3_filename = ''
ils_cluster3_fp = ''
ils_cluster4_filename = ''
ils_cluster4_fp = ''

il_clustera_model = ''
il_cluster0_model = ''
il_cluster1_model = ''
il_cluster2_model = ''
il_cluster3_model = ''
il_cluster4_model = ''

report_filename = ''
report_fp = ''

ils_dagger_iter = ''

##### End of Added for ILS

config = configparser.ConfigParser()
config.read('config_file.ini')

# Assign debug variable to be true to check the flow of the program
DEBUG_CONFIG = config.getboolean('DEBUG',
                                 'debug_config')  # Debug variable to check the DASH-Sim configuration related debug messages
DEBUG_SIM = config.getboolean('DEBUG',
                              'debug_sim')  # Debug variable to check the Simulation core related debug messages
DEBUG_JOB = config.getboolean('DEBUG', 'debug_job')  # Debug variable to check the Job generator related debug messages
DEBUG_SCH = config.getboolean('DEBUG', 'debug_sch')  # Debug variable to check the Scheduler related debug messages

# Assign info variable to be true to get the information about the flow of the program
INFO_SIM = config.getboolean('INFO', 'info_sim')  # Info variable to check the Simulation core related info messages
INFO_JOB = config.getboolean('INFO', 'info_job')  # Info variable to check the job generator related info messages
INFO_SCH = config.getboolean('INFO', 'info_sch')  # Info variable to check the Scheduler related info messages

# Assign scheduler name variable
scheduler = config['DEFAULT']['scheduler']
lam = float(config['DEFAULT']['lam'])
theta_1 = float(config['DEFAULT']['theta_1'])
# Assign trace variable to be true to save traces from the execution
CLEAN_TRACES = config.getboolean('TRACE', 'clean_traces')  # Flag used to clear previous traces
TRACE_TASKS = config.getboolean('TRACE', 'trace_tasks')  # Trace information from each task
TRACE_SYSTEM = config.getboolean('TRACE', 'trace_system')  # Trace information from the whole system
TRACE_FREQUENCY = config.getboolean('TRACE', 'trace_frequency')  # Trace frequency variation information
TRACE_PES = config.getboolean('TRACE', 'trace_PEs')  # Trace information from each PE
TRACE_IL_PREDICTIONS = config.getboolean('TRACE', 'trace_IL_predictions')  # Trace the predictions of the IL policy
TRACE_TEMPERATURE = config.getboolean('TRACE', 'trace_temperature')  # Trace temperature information
TRACE_LOAD = config.getboolean('TRACE', 'trace_load')  # Trace system load information
CREATE_DATASET_DTPM = config.getboolean('TRACE', 'create_dataset_DTPM')  # Create dataset for the ML algorithm
TRACE_FILE_TASKS = config['TRACE']['trace_file_tasks']  # Trace file name for the task trace
TRACE_FILE_SYSTEM = config['TRACE']['trace_file_system']  # Trace file name for the system trace
TRACE_FILE_FREQUENCY = config['TRACE']['trace_file_frequency']  # Trace file name for the frequency trace
TRACE_FILE_PES = config['TRACE']['trace_file_PEs']  # Trace file name for the PE trace
TRACE_FILE_IL_PREDICTIONS = config['TRACE']['trace_file_IL_predictions']  # Trace file name for the IL predictions
TRACE_FILE_TEMPERATURE = config['TRACE']['trace_file_temperature']  # Trace file name for the temperature trace
TRACE_FILE_TEMPERATURE_WORKLOAD = config['TRACE'][
    'trace_file_temperature_workload']  # Trace file name for the temperature trace (workload)
TRACE_FILE_LOAD = config['TRACE']['trace_file_load']  # Trace file name for the load trace
HARDWARE_COUNTERS_TRACE = config['TRACE']['hardware_counters_trace']  # Trace file name for the hardware counters
HARDWARE_COUNTERS_SINGLE_APP = config['TRACE'][
    'hardware_counters_single_app']  # Trace file name for the hardware counters (single application)
RESULTS = config['TRACE'][
    'results']  # Trace file name for the results of the simulation, including exec time, energy, etc.
DATASET_FILE_DTPM = config['TRACE']['dataset_file_DTPM']  # Dataset file name for the ML algorithm
DEADLINE_FILE = config['TRACE']['deadline_file']  # File that contains the deadlines for each snippet
TRACE_WIFI_TX = config['TRACE']['trace_wifi_TX']  # Hardware counter trace for WiFi TX
TRACE_WIFI_RX = config['TRACE']['trace_wifi_RX']  # Hardware counter trace for WiFi RX
TRACE_RANGE_DET = config['TRACE']['trace_range_det']  # Hardware counter trace for Range Detection
TRACE_SCT = config['TRACE']['trace_SCT']  # Hardware counter trace for SCT
TRACE_SCR = config['TRACE']['trace_SCR']  # Hardware counter trace for SCR
TRACE_TEMP_MIT = config['TRACE']['trace_TEMP_MIT']  # Hardware counter trace for Temporal Mitigation
MERGE_LIST = [TRACE_WIFI_TX, TRACE_WIFI_RX, TRACE_RANGE_DET, TRACE_SCT, TRACE_SCR,
              TRACE_TEMP_MIT]  # Configure this parameter to merge different traces

generate_complete_trace = False  # This flag is set by generate_traces.py and DTPM_run_dagger.py, do not modify
DVFS_cfg_list = []  # This flag is set by generate_traces.py and DTPM_run_dagger.py, do not modify
num_PEs_TRACE = 0  # This flag is set by generate_traces.py and DTPM_run_dagger.py, do not modify
trace_file_num = 0  # This flag is set by generate_traces.py and DTPM_run_dagger.py, do not modify
gen_trace_capacity_little = -1  # This flag is set by generate_traces.py and DTPM_run_dagger.py, do not modify
gen_trace_capacity_big = -1  # This flag is set by generate_traces.py and DTPM_run_dagger.py, do not modify
sim_ID = 0

seed = int(config['DEFAULT']['random_seed'])  # Specify a seed value for the random number generator
simulation_clk = int(config['DEFAULT']['clock'])  # The core simulation engine tick with simulation_clk
simulation_length = int(config['DEFAULT']['simulation_length'])  # The length of the simulation (in us)
simulation_num = int(config['DEFAULT']['simulation_num'])  # The length of the simulation (in us)
standard_deviation = float(
    config['DEFAULT']['standard_deviation'])  # Standard deviation for randomization of execution time
use_adaptive_scheduling = config.getboolean('SCHEDULER PARAMETERS',
                                            'heft_adaptive')  # Whether scheduling should seek to be adaptive between makespan and EDP priorities

# IL Scheduler Parameters
# Specify if dataset is to be saved 
ils_enable_dataset_save = config.getboolean('IL SCHEDULER', 'enable_dataset_save')
# Specify if IL policy should be used for scheduling decisions
ils_enable_policy_decision = config.getboolean('IL SCHEDULER', 'enable_ils_policy')
# Specify if IL DAgger is performed
ils_enable_dagger = config.getboolean('IL SCHEDULER', 'enable_ils_dagger')
# Specify the regression tree depth of the generated models
ils_RT_tree_depth = int(config['IL SCHEDULER']['RT_tree_depth'])

# POWER MANAGEMENT
sampling_rate = int(config['POWER MANAGEMENT']['sampling_rate'])  # Specify the sampling rate for the DVFS mechanism
sampling_rate_temperature = int(
    config['POWER MANAGEMENT']['sampling_rate_temperature'])  # Specify the sampling rate for the temperature update
snippet_size = int(config['POWER MANAGEMENT']['snippet_size'])  # Specify the snippet size
util_high_threshold = float(
    config['POWER MANAGEMENT']['util_high_threshold'])  # Specify the high threshold (ondemand mode)
util_low_threshold = float(
    config['POWER MANAGEMENT']['util_low_threshold'])  # Specify the low threshold  (ondemand mode)
DAgger_iter = int(config['POWER MANAGEMENT']['DAgger_iter'])  # Number of iterations from the DAgger algorithm
ml_algorithm = config['POWER MANAGEMENT']['ml_algorithm']  # Specify the machine learning-based policy for DTPM
optimization_objective = config['POWER MANAGEMENT']['optimization_objective']  # Specify the optimization objective
enable_real_time_constraints = config.getboolean('POWER MANAGEMENT',
                                                 'enable_real_time_constraints')  # Flag to enable snippet deadlines to be considered in DTPM
real_time_aware_oracle = config.getboolean('POWER MANAGEMENT',
                                           'real_time_aware_oracle')  # Flag to enable real-time constraints in the oracle generation
enable_regression_policy = config.getboolean('POWER MANAGEMENT',
                                             'enable_regression_policy')  # Flag to enable the regression policy for real-time constraints
enable_num_cores_prediction = config.getboolean('POWER MANAGEMENT',
                                                'enable_num_cores_prediction')  # Flag to enable the prediction of the optimal number of cores per cluster
enable_thermal_management = config.getboolean('POWER MANAGEMENT',
                                              'enable_thermal_management')  # Flag to enable the thermal managent
train_on_reduced_dataset = config.getboolean('POWER MANAGEMENT',
                                             'train_on_reduced_dataset')  # Flag to enable training on the reduced dataset (uses complete dataset if False)
remove_app_ID = int(
    config['POWER MANAGEMENT']['remove_app_ID'])  # Specify the app ID to be removed from the reduced dataset
DTPM_thermal_limit = int(config['POWER MANAGEMENT'][
                             'DTPM_thermal_limit'])  # Specify the thermal limit that the thermal managament policy should consider
N_steps_temperature_prediction = int(config['POWER MANAGEMENT'][
                                         'N_steps_temperature_prediction'])  # Specify the number of steps for the temperature prediction
DTPM_freq_policy_file = config['POWER MANAGEMENT']['DTPM_freq_policy_file']  # Specify the filename for the DTPM policy
DTPM_num_cores_policy_file = config['POWER MANAGEMENT'][
    'DTPM_num_cores_policy_file']  # Specify the filename for the DTPM policy
DTPM_regression_policy_file = config['POWER MANAGEMENT'][
    'DTPM_regression_file']  # Specify the filename for the DTPM policy
enable_throttling = config.getboolean('POWER MANAGEMENT', 'enable_throttling')  # Flag to enable the thermal throttling
enable_DTPM_throttling = config.getboolean('POWER MANAGEMENT',
                                           'enable_DTPM_throttling')  # Flag to enable the thermal throttling for the custom DTPM policies
C1 = float(config['POWER MANAGEMENT']['C1'])  # Coefficient for the leakage model
C2 = int(config['POWER MANAGEMENT']['C2'])  # Coefficient for the leakage model
Igate = float(config['POWER MANAGEMENT']['Igate'])  # Coefficient for the leakage model
T_ambient = float(config['POWER MANAGEMENT']['T_ambient'])  # Ambient temperature

# SIMULATION MODE
simulation_mode = config['SIMULATION MODE']['simulation_mode']  # Defines under which mode, the simulation will be run
inject_jobs_ASAP = config.getboolean('SIMULATION MODE', 'inject_jobs_asap')
sim_early_stop = config.getboolean('SIMULATION MODE', 'sim_early_stop')
if simulation_mode not in ('validation', 'performance'):
    print('[E] Please choose a valid simulation mode')
    print(simulation_mode)
    sys.exit()
max_jobs_in_parallel = int(config['SIMULATION MODE']['max_jobs_in_parallel'])
# variables used under performance mode
warmup_period = int(
    config['SIMULATION MODE']['warmup_period'])  # is the time period till which no result will be recorded
num_of_iterations = int(
    config['SIMULATION MODE']['num_of_iterations'])  # The number of iteration at each job injection rate
fixed_injection_rate = config.getboolean('SIMULATION MODE', 'fixed_injection_rate')

tasks_dict = {}
jobID = {}
baseID = {}
prednode = {}
p = config['DEFAULT']['p'].split(',')
for i in range(len(p)):
    p[i] = float(p[i])
running = []
exe_time_tmp = []
task_schedules = {}

deadline_type = config['DEFAULT']['deadline_type']
resource_file = config['DEFAULT']['resource_file']
exe_time = 0
task_num = 0
a = 0
deadline = 0
num_of_out = 0
num_of_out_1 = 0
gg = -1
num_of_jobs = 0
num_of_jobs_1 = 0
sele = 0
cost = 0
end = []
acj = 0
acj_sum = 0
nums = 0
job_name_temp = ''
np.random.seed(50)
rand = np.random.random()
curr_dag = nx.DiGraph()
num_node = 0
times = 0
communication_dict = {}
computation = {}
power = {}
deadline_dict = {}
arrive_time = {}
proc_schedule = {}
offset = 0
num_of_jobs_same_time = 0
job_statistics = {}
overtime_job_statistics = {}
overtime_sum = 0
jobname = ''
overtime_job_time_statistics = {}
yield_time_1 = random.poisson(lam=50, size=408)
yield_time_2 = random.poisson(lam=50, size=2028)
yield_turn = 0
texit = {}

proc_schedule_1 = []
nodes = {}
t = {}
t_1 = {}
FT = {}
P = {}
running_task = {}
jobID_now = 0
p_p = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
p_num = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
flag_1 = False
# for T
job_dag_t = -1
merged_dag_t = -1
PEs_t = -1
ready_queue_t = -1
computation_matrix_t = -1
job_ID_t = -1
proc_schedule_t = -1
time_offset_t = -1
relabel_nodes_t = -1
power_dict_t = -1
rank_metric_t = -1
task_list_t = -1
offset_t = -1
table_t = -1
sorted_nodes = -1
out_dag = -1
struct_self = -1
tmp_ec = 100
s = 0


def str_to_list(x):
    # function to return a list based on a formatted string
    result = []
    if '[' in x:
        result = ast.literal_eval(x)
    else:
        for part in x.split(','):
            if ('-' in part):
                a, b, c = part.split('-')
                a, b, c = int(a), int(b), int(c)
                result.extend(range(a, b, c))
            elif ('txt' in part):
                result.append(part)
            else:
                a = int(part)
                result.append(a)
    return result


# end of def str_to_list(x)

trip_temperature = str_to_list(config['POWER MANAGEMENT']['trip_temperature'])  # List of temperature trip points
trip_hysteresis = str_to_list(config['POWER MANAGEMENT']['trip_hysteresis'])  # List of hysteresis trip points
DTPM_trip_temperature = str_to_list(
    config['POWER MANAGEMENT']['DTPM_trip_temperature'])  # List of temperature trip points for the custom DTPM policies

config_scale_values = config['SIMULATION MODE']['scale_values']
if not (ils_enable_policy_decision or ils_enable_dataset_save):
    scale_values_list = str_to_list(
        config_scale_values)  # List of scale values which will determine the job arrival rate under performance mode
## if not (ils_enable_policy_decision or ils_enable_dataset_save) :


job_list = str_to_list(config['SIMULATION MODE'][
                           'job_list'])  # List containing the number of jobs that should be executed for each application
if len(job_list) > 0:
    current_job_list = job_list[0]
else:
    current_job_list = []
job_counter_list = [0] * len(current_job_list)  # List to count the number of injected jobs for each application

if len(job_list) > 0:
    max_num_jobs = int(config['SIMULATION MODE']['jobs']) * len(job_list)
else:
    max_num_jobs = int(config['SIMULATION MODE']['jobs'])
scale = int(
    config['SIMULATION MODE']['scale'])  # The variable used to adjust the mean value of the job inter-arrival time
if (simulation_mode == 'validation'):
    warmup_period = 0  # Warmup period is zero under validation mode

# COMMUNICATION MODE
PE_to_PE = config.getboolean('COMMUNICATION MODE',
                             'PE_to_PE')  # The communication mode in which data is sent, directly, from a PE to a PE
shared_memory = config.getboolean('COMMUNICATION MODE',
                                  'shared_memory')  # The communication mode in which data is sent from a PE to a PE through a shared memory

if (PE_to_PE) and (shared_memory):
    print('[E] Please chose only one of the communication modes')
    sys.exit()
elif (not PE_to_PE) and (not shared_memory):
    print('[E] Please chose one of the communication modes')
    sys.exit()

# The variables used by table-based schedulers
table = {}
table_1 = {}
table_t = {}
table_2 = -1
table_3 = -1
table_4 = -1
temp_list = []
# Additional variables used by list-based schedulers
current_dag = nx.DiGraph()
computation_dict = {}
power_dict = {}

# DTPM
IL_freq_policy = None
IL_num_cores_policy = None
IL_regression_policy = None
current_temperature_vector = [T_ambient,  # Indicate the current PE temperature for each hotspot
                              T_ambient,
                              T_ambient,
                              T_ambient,
                              T_ambient]
B_model = []
throttling_state = -1

# Snippet_inj is incremented every time a snippet finishes being injected
snippet_ID_inj = -1
# Snippet_exec is incremented every time a snippet finishes being executed
snippet_ID_exec = 0
snippet_throttle = -1
snippet_temp_list = []
snippet_initial_temp = [T_ambient,
                        T_ambient,
                        T_ambient,
                        T_ambient,
                        T_ambient]
DAgger_last_snippet_ID_freq = -1
DAgger_last_snippet_ID_num_cores = -1
DAgger_last_snippet_ID_regression = -1
first_DAgger_iteration = False

snippet_start_time = 0
oracle_config_dict = {}
deadline_dict = {}
total_predictions = 0
wrong_predictions_freq = 0
wrong_predictions_num_cores = 0
missed_deadlines = 0
hardware_counters = {}
aggregate_data_freq = False
aggregate_data_num_cores = False
aggregate_data_regression = False
thermal_limit_violated = False
previous_freq = []
previous_num_cores = []


class PerfStatics:
    '''
    Define the PerfStatics class to calculate power consumption and total execution time.
    '''

    def __init__(self):
        self.execution_time = 0.0  # The total execution time (us)
        self.energy_consumption = 0.0  # The total energy consumption (uJ)
        self.cumulative_exe_time = 0.0  # Sum of the execution time of completed tasks (us)
        self.cumulative_exe_time_1 = 0.0  # Sum of the execution time of completed tasks (us)
        self.cumulative_energy_consumption = 0.0  # Sum of the energy consumption of completed tasks (us)
        self.injected_jobs = 0  # Count the number of jobs that enter the system (i.e. the ready queue)
        self.completed_jobs = 0  # Count the number of jobs that are completed
        self.ave_execution_time = 0.0  # Average execution time for the jobs that are finished
        self.job_counter = 0  # Indicate the number of jobsin the system at any given time
        self.average_job_number = 0  # Shows the average number of jobs in the system for a workload
        self.job_counter_list = []
        self.sampling_rate_list = []
        # more


# end class PerfStatics

# Instantiate the object that will store the performance statistics
global results


class Validation:
    '''
    Define the Validation class to compare the generated and completed jobs
    '''
    start_times = []
    finish_times = []
    generated_jobs = []
    injected_jobs = []
    completed_jobs = []


# end class Validation

class Resource:
    '''
    Define the Resource class to define a resource
    It stores properties of the resources.
	'''

    def __init__(self):
        self.type = ''  # The type of the resource (CPU, FFT_ACC, etc.)
        self.name = ''  # Name of the resource
        self.ID = -1  # This is the unique ID of the resource. "-1" means it is not initialized
        self.cluster_ID = -1  # ID of the cluster this PE belongs to
        self.capacity = 1  # Number tasks that a resource can run simultaneously. Default value is 1.
        self.num_of_functionalites = 0  # This variable shows how many different task this resource can run
        self.supported_functionalities = []  # List of all tasks can be executed by Resource
        self.performance = []  # List of runtime (in micro seconds) for each supported task
        self.idle = True  # initial state of Resource which idle and ready for a task (normalized to the number of instructions)
        self.cost = 0


# end class Resource

class ResourceManager:
    '''
    Define the ResourceManager class to maintain
    the list of the resource in our DASH-SoC model.
    '''

    def __init__(self):
        # 可用资源
        self.list = []  # list of available resources
        # communication consumption矩阵
        self.comm_band = []  # This variable represents the communication bandwidth matrix


# end class ResourceManager

class ClusterManager:
    '''
    Define the ClusterManager class to maintain
    the list of clusters in our DASH-SoC model.
    '''

    cluster_list = None

    def __init__(self):
        self.cluster_list = []  # list of available clusters


# end class ClusterManager

class Tasks:
    '''
    Define the Tasks class to maintain the list
    of tasks. It stores properties of the tasks.
    '''

    def __init__(self):
        self.name = ''  # The name of the task
        # 加了offset
        self.ID = -1  # This is the unique ID of the task. "-1" means it is not initialized
        self.predecessors = []  # List of all task IDs to identify task dependency
        #################################################################
        ## Code added by Anish to dump IL features
        #################################################################
        self.preds = []  # List of all task IDs to identify task dependency
        self.il_features = []  # Min, max, mean, median and variance of exec. times across resources
        self.dag_depth = -1  # The downward depth of current task to end of DAG
        self.weight = []  # Dynamic weight of task
        self.head_ID = -1  # ID of root node
        #################################################################
        ## End of Code added by Anish to dump IL features
        #################################################################
        self.est = -1  # This variable represents the earliest time that a task can start
        self.deadline = -1  # This variable represents the deadline for a task
        self.head = False  # If head is true, this task is the leading (the first) element in a task graph
        self.tail = False  # If tail is true, this task is the end (the last) element in a task graph
        self.jobID = -1  # This task belongs to job with this ID
        self.jobname = ''  # This task belongs to job with this name
        # 没加offset
        self.base_ID = -1  # This ID will be used to calculate the data volume from one task to another
        self.PE_ID = -1  # Holds the PE ID on which the task will be executed
        self.start_time = -1  # Execution start time of a task
        self.finish_time = -1  # Execution finish time of a task
        self.order = -1  # Relative ordering of this task on a particular PE
        self.dynamic_dependencies = []  # List of dynamic dependencies that a scheduler requests are satisfied before task launch
        self.ready_wait_times = []  # List holding wait times for a task for being ready due to communication time from its predecessor
        self.execution_wait_times = []  # List holding wait times for a task for being execution-ready due to communication time between memory and a PE
        self.PE_to_PE_wait_time = []  # List holding wait times for a task for being execution-ready due to PE to PE communication time
        self.order = -1  # Execution order if a list based scheduler is used, e.g., ILP
        self.task_elapsed_time_max_freq = 0  # Indicate the current execution time for a given task
        self.job_start = -1  # Holds the execution start time of a head task (also execution start time for a job)
        self.time_stamp = -1  # This values used to check whether all data for the task is transferred or not
        self.st = -1
        self.isChange = False


# end class Tasks

class TaskManager:
    '''
    Define the TaskManager class to maintain the
    list of the tasks in our DASH-SoC model.
    '''

    def __init__(self):
        self.list = []  # List of available tasks
        self.set = set()


# end class TaskManager

class Applications:
    '''
    Define the Applications class to maintain the
    all information about an application (job)
    '''

    def __init__(self):
        self.name = ''  # The name of the application
        self.task_list = []  # List of all tasks in an application
        self.comm_vol = []  # This variable represents the communication volume matrix
        # i.e. each entry is data volume should be transferred from one task to another


# end class Applications

class ApplicationManager:
    '''
    Define the ApplicationManager class to maintain the
    list of the applications (jobs) in our DASH-SoC model.
    '''

    def __init__(self):
        self.list = []  # List of all applications


# end class ApplicationManager

class TaskQueues:
    '''
    Define the TaskQueues class to maintain the
    all task queue lists
    '''

    def __init__(self):
        self.outstanding = []  # List of *all* tasks waiting to be processed
        self.ready = []  # List of tasks that are ready for processing
        self.running = []  # List of currently running tasks
        self.completed = []  # List of completed tasks
        self.wait_ready = []  # List of task waiting for being pushed into ready queue because of memory communication time
        self.executable = []  # List of task waiting for being executed because of memory communication time


# end class TaskQueues

# =============================================================================
# def clear_screen():
#     '''
#     Define the clear_screen function to
#     clear the screen before the simulation.
#     '''
#     current_platform = platform.system()        # Find the platform
#     if 'windows' in current_platform.lower():
#         get_ipython().magic('clear')
#     elif 'Darwin' in current_platform.lower():
#         get_ipython().magic('clear')
#     elif 'linux' in current_platform.lower():
#         get_ipython().magic('clear')  
# # end of def clear_screen()
# =============================================================================

DyPO_prev_index = -1
DyPO_prev_index_1 = -1
DyPO_dypo_prev_freq = 0
DyPO_dypo_curr_freq = 0
DyPO_dypo_n_big = 2
DyPO_dypo_req_big = 2
DyPO_STATUS = 0
DyPO_F_STATUS = 0


def get_cluster(resource):
    if resource >= 0 and resource <= 3:
        cluster = 0
    elif resource >= 4 and resource <= 7:
        cluster = 1
    elif resource >= 8 and resource <= 9:
        cluster = 2
    elif resource >= 10 and resource <= 13:
        cluster = 3
    elif resource >= 14 and resource <= 15:
        cluster = 4
    else:
        cluster = -1
    return cluster


def get_supported_clusters(task_ID, job_type, cluster):
    if job_type == 0:
        cluster_supported_functionalities = []
        cluster_supported_functionalities.append(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
        cluster_supported_functionalities.append(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
        cluster_supported_functionalities.append([0])
        cluster_supported_functionalities.append([4, 9, 14, 19, 24])
        cluster_supported_functionalities.append([])

        if task_ID in cluster_supported_functionalities[cluster]:
            return 0
        else:
            return -1

    elif job_type == 1:
        cluster_supported_functionalities = []
        cluster_supported_functionalities.append(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
        cluster_supported_functionalities.append(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
        cluster_supported_functionalities.append([0])
        cluster_supported_functionalities.append([4, 9, 14, 19, 24])
        cluster_supported_functionalities.append([])

        if task_ID in cluster_supported_functionalities[cluster]:
            return 0
        else:
            return -1

    else:
        print('[ERROR] Incorrect job type specified. Exiting simulation...')
        sys.exit(0)


def get_normalized_list(input_list):
    ## Convert list to numpy array
    np_input_list = np.array(input_list)

    ## Check for list of all zeros
    if len(np_input_list[np_input_list != 0]) == 0:
        return input_list
    ## if len(np_input_list[np_input_list != 0]) == 0 :

    ## Exclude anomalous values
    valid_value_list = np_input_list[np_input_list != 10000]

    ## Check exception and return if list is empty
    if len(valid_value_list) == 0:
        valid_value_list = np.ones(len(input_list) * 10)
        return valid_value_list
    ## if len(valid_value_list) :

    ## Get min and range of values in list
    min_value_list = np.min(valid_value_list)
    range_value_list = np.max(valid_value_list) - np.min(valid_value_list)

    ## Check exception and normalize
    if range_value_list == 0:
        normalized_list = np_input_list / min_value_list
    else:
        normalized_list = (np_input_list - min_value_list) / range_value_list
    ## if range_value_list == 0 :

    ## If value is greater than one
    normalized_list[normalized_list > 1] = 10

    return normalized_list


## def get_normalized_list(input_list) :

def ils_print_file_headers():
    global ils_clustera_filename_header
    global ils_clustera_filename
    global ils_clustera_fp
    global ils_cluster0_filename
    global ils_cluster0_fp
    global ils_cluster1_filename
    global ils_cluster1_fp
    global ils_cluster2_filename
    global ils_cluster2_fp
    global ils_cluster3_filename
    global ils_cluster3_fp
    global ils_cluster4_filename
    global ils_cluster4_fp

    if ils_clustera_filename_header == 0:
        ils_clustera_fp.write('Time,')
        ils_clustera_fp.write('TaskID,')
        for value in range(5):
            ils_clustera_fp.write('Cluster' + str(value) + '_FreeTime,')
        ## for value in range(5) :
        ils_clustera_fp.write('NormTaskID,')
        for value in range(5):
            ils_clustera_fp.write('ExecTime_Cluster' + str(value) + ',')
        ## for value in range(5) :
        ils_clustera_fp.write('DownwardDepth,')
        ils_clustera_fp.write('RelativeJobID,')
        ils_clustera_fp.write('JobType,')
        for value in range(5):
            ils_clustera_fp.write('Pred' + str(value) + '_ID,')
        ## for value in range(5) :
        for value in range(5):
            ils_clustera_fp.write('Pred' + str(value) + '_Cluster,')
        ## for value in range(5) :
        for value in range(5):
            ils_clustera_fp.write('Pred' + str(value) + '_Comm,')
        ## for value in range(5) :
        ils_clustera_fp.write('PE_label,Cluster_label\n')

        ils_cluster0_fp.write('Time,')
        ils_cluster0_fp.write('TaskID,')
        for value in range(5):
            ils_cluster0_fp.write('PE' + str(value) + '_FreeTime,')
        ## for value in range(5) :
        ils_cluster0_fp.write('NormTaskID,')
        for value in range(5):
            ils_cluster0_fp.write('ExecTime_Cluster' + str(value) + ',')
        ## for value in range(5) :
        ils_cluster0_fp.write('DownwardDepth,')
        ils_cluster0_fp.write('RelativeJobID,')
        ils_cluster0_fp.write('JobType,')
        for value in range(5):
            ils_cluster0_fp.write('Pred' + str(value) + '_ID,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster0_fp.write('Pred' + str(value) + '_Cluster,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster0_fp.write('Pred' + str(value) + '_Comm,')
        ## for value in range(5) :
        ils_cluster0_fp.write('PE_label,Cluster_label\n')

        ils_cluster1_fp.write('Time,')
        ils_cluster1_fp.write('TaskID,')
        for value in range(5):
            ils_cluster1_fp.write('PE' + str(value) + '_FreeTime,')
        ## for value in range(5) :
        ils_cluster1_fp.write('NormTaskID,')
        for value in range(5):
            ils_cluster1_fp.write('ExecTime_Cluster' + str(value) + ',')
        ## for value in range(5) :
        ils_cluster1_fp.write('DownwardDepth,')
        ils_cluster1_fp.write('RelativeJobID,')
        ils_cluster1_fp.write('JobType,')
        for value in range(5):
            ils_cluster1_fp.write('Pred' + str(value) + '_ID,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster1_fp.write('Pred' + str(value) + '_Cluster,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster1_fp.write('Pred' + str(value) + '_Comm,')
        ## for value in range(5) :
        ils_cluster1_fp.write('PE_label,Cluster_label\n')

        ils_cluster2_fp.write('Time,')
        ils_cluster2_fp.write('TaskID,')
        for value in range(5):
            ils_cluster2_fp.write('PE' + str(value) + '_FreeTime,')
        ## for value in range(5) :
        ils_cluster2_fp.write('NormTaskID,')
        for value in range(5):
            ils_cluster2_fp.write('ExecTime_Cluster' + str(value) + ',')
        ## for value in range(5) :
        ils_cluster2_fp.write('DownwardDepth,')
        ils_cluster2_fp.write('RelativeJobID,')
        ils_cluster2_fp.write('JobType,')
        for value in range(5):
            ils_cluster2_fp.write('Pred' + str(value) + '_ID,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster2_fp.write('Pred' + str(value) + '_Cluster,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster2_fp.write('Pred' + str(value) + '_Comm,')
        ## for value in range(5) :
        ils_cluster2_fp.write('PE_label,Cluster_label\n')

        ils_cluster3_fp.write('Time,')
        ils_cluster3_fp.write('TaskID,')
        for value in range(5):
            ils_cluster3_fp.write('PE' + str(value) + '_FreeTime,')
        ## for value in range(5) :
        ils_cluster3_fp.write('NormTaskID,')
        for value in range(5):
            ils_cluster3_fp.write('ExecTime_Cluster' + str(value) + ',')
        ## for value in range(5) :
        ils_cluster3_fp.write('DownwardDepth,')
        ils_cluster3_fp.write('RelativeJobID,')
        ils_cluster3_fp.write('JobType,')
        for value in range(5):
            ils_cluster3_fp.write('Pred' + str(value) + '_ID,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster3_fp.write('Pred' + str(value) + '_Cluster,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster3_fp.write('Pred' + str(value) + '_Comm,')
        ## for value in range(5) :
        ils_cluster3_fp.write('PE_label,Cluster_label\n')

        ils_cluster4_fp.write('Time,')
        ils_cluster4_fp.write('TaskID,')
        for value in range(5):
            ils_cluster4_fp.write('PE' + str(value) + '_FreeTime,')
        ## for value in range(5) :
        ils_cluster4_fp.write('NormTaskID,')
        for value in range(5):
            ils_cluster4_fp.write('ExecTime_Cluster' + str(value) + ',')
        ## for value in range(5) :
        ils_cluster4_fp.write('DownwardDepth,')
        ils_cluster4_fp.write('RelativeJobID,')
        ils_cluster4_fp.write('JobType,')
        for value in range(5):
            ils_cluster4_fp.write('Pred' + str(value) + '_ID,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster4_fp.write('Pred' + str(value) + '_Cluster,')
        ## for value in range(5) :
        for value in range(5):
            ils_cluster4_fp.write('Pred' + str(value) + '_Comm,')
        ## for value in range(5) :
        ils_cluster4_fp.write('PE_label,Cluster_label\n')

        ils_clustera_filename_header = 1
    ## if ils_clustera_filename_header == 0 :


## def ils_print_file_headers() :

def ils_setup(scale_values):
    global ils_clustera_filename_header
    global ils_clustera_filename
    global ils_clustera_fp
    global ils_cluster0_filename
    global ils_cluster0_fp
    global ils_cluster1_filename
    global ils_cluster1_fp
    global ils_cluster2_filename
    global ils_cluster2_fp
    global ils_cluster3_filename
    global ils_cluster3_fp
    global ils_cluster4_filename
    global ils_cluster4_fp

    global ils_enable_policy_decision
    global ils_enable_dataset_save

    global il_clustera_model
    global il_cluster0_model
    global il_cluster1_model
    global il_cluster2_model
    global il_cluster3_model
    global il_cluster4_model

    global ils_RT_tree_depth
    global ils_dagger_iter

    global report_filename
    global report_fp

    dagger_string = ''
    model_dagger_string = ''
    if ils_dagger_iter != '':
        if int(ils_dagger_iter) != 1:
            model_dagger_string = '_dagger' + str(int(ils_dagger_iter) - 1)
        dagger_string = '_dagger' + str(int(ils_dagger_iter))

    if ils_enable_dataset_save or ils_enable_dagger:
        os.makedirs('./reports', exist_ok=True)
        os.makedirs('./datasets', exist_ok=True)

        ils_clustera_filename = './datasets/data_IL_clustera_' + scale_values + dagger_string + '.csv'
        ils_cluster0_filename = './datasets/data_IL_cluster0_' + scale_values + dagger_string + '.csv'
        ils_cluster1_filename = './datasets/data_IL_cluster1_' + scale_values + dagger_string + '.csv'
        ils_cluster2_filename = './datasets/data_IL_cluster2_' + scale_values + dagger_string + '.csv'
        ils_cluster3_filename = './datasets/data_IL_cluster3_' + scale_values + dagger_string + '.csv'
        ils_cluster4_filename = './datasets/data_IL_cluster4_' + scale_values + dagger_string + '.csv'

        ils_clustera_fp = open(ils_clustera_filename, 'w')
        ils_cluster0_fp = open(ils_cluster0_filename, 'w')
        ils_cluster1_fp = open(ils_cluster1_filename, 'w')
        ils_cluster2_fp = open(ils_cluster2_filename, 'w')
        ils_cluster3_fp = open(ils_cluster3_filename, 'w')
        ils_cluster4_fp = open(ils_cluster4_filename, 'w')
    ## if ils_enable_dataset_save :

    if ils_enable_policy_decision:
        il_clustera_model = pickle.load(
            open('./models/RT_clustera_merged' + model_dagger_string + '_model_' + str(ils_RT_tree_depth) + '.sav',
                 'rb'))
        il_cluster0_model = pickle.load(
            open('./models/RT_cluster0_merged' + model_dagger_string + '_model_' + str(ils_RT_tree_depth) + '.sav',
                 'rb'))
        il_cluster1_model = pickle.load(
            open('./models/RT_cluster1_merged' + model_dagger_string + '_model_' + str(ils_RT_tree_depth) + '.sav',
                 'rb'))
        il_cluster2_model = pickle.load(
            open('./models/RT_cluster2_merged' + model_dagger_string + '_model_' + str(ils_RT_tree_depth) + '.sav',
                 'rb'))
        il_cluster3_model = pickle.load(
            open('./models/RT_cluster3_merged' + model_dagger_string + '_model_' + str(ils_RT_tree_depth) + '.sav',
                 'rb'))
        il_cluster4_model = pickle.load(
            open('./models/RT_cluster4_merged' + model_dagger_string + '_model_' + str(ils_RT_tree_depth) + '.sav',
                 'rb'))
    ## if ils_enable_policy_decision :


## def ils_open_file_handles() :

def open_report_file_handles(scale_values):
    global ils_dagger_iter

    global report_filename
    global report_fp
    global scheduler

    dagger_string = ''
    model_dagger_string = ''
    if ils_dagger_iter != '':
        if int(ils_dagger_iter) != 1:
            model_dagger_string = '_dagger' + str(int(ils_dagger_iter) - 1)
        dagger_string = '_dagger' + str(int(ils_dagger_iter))

    if ils_enable_dataset_save or ils_enable_dagger:
        os.makedirs('./reports', exist_ok=True)
        os.makedirs('./datasets', exist_ok=True)

    # Create file handles
    report_filename = './reports/report_' + scheduler + '_' + scale_values + model_dagger_string + '.rpt'
    report_fp = open(report_filename, 'w')


def ils_open_file_handles():
    global ils_clustera_filename_header
    global ils_clustera_filename
    global ils_clustera_fp
    global ils_cluster0_filename
    global ils_cluster0_fp
    global ils_cluster1_filename
    global ils_cluster1_fp
    global ils_cluster2_filename
    global ils_cluster2_fp
    global ils_cluster3_filename
    global ils_cluster3_fp
    global ils_cluster4_filename
    global ils_cluster4_fp

    ils_clustera_fp = open(ils_clustera_filename, 'a')
    ils_cluster0_fp = open(ils_cluster0_filename, 'a')
    ils_cluster1_fp = open(ils_cluster1_filename, 'a')
    ils_cluster2_fp = open(ils_cluster2_filename, 'a')
    ils_cluster3_fp = open(ils_cluster3_filename, 'a')
    ils_cluster4_fp = open(ils_cluster4_filename, 'a')


## def ils_open_file_handles() :

def ils_close_file_handles():
    global ils_clustera_filename_header
    global ils_clustera_filename
    global ils_clustera_fp
    global ils_cluster0_filename
    global ils_cluster0_fp
    global ils_cluster1_filename
    global ils_cluster1_fp
    global ils_cluster2_filename
    global ils_cluster2_fp
    global ils_cluster3_filename
    global ils_cluster3_fp
    global ils_cluster4_filename
    global ils_cluster4_fp

    ils_clustera_fp.close()
    ils_cluster0_fp.close()
    ils_cluster1_fp.close()
    ils_cluster2_fp.close()
    ils_cluster3_fp.close()
## def ils_close_file_handles() :
