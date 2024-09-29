'''
Description: This file is the main() function which should be run to get the stimulation results.
'''
import simpy
import configparser
import argparse
import matplotlib.pyplot as plt                                                 
import random                                                                  
import numpy as np
import sys
import os
import pandas as pd
import networkx as nx
from shutil import copyfile
import pickle
import csv

import heft.gantt
import heftrt.gantt
import job_generator                                                            # Dynamic job generation is handled by job_generator.py
import common                                                                   # The common parameters used in DASH-Sim are defined in common.py
import DASH_SoC_parser                                                          # The resource parameters used in DASH-Sim are obtained from
                                                                                # Resource initialization file(DASH.SoC.**.txt), parsed by DASH_SoC_parser.py
import job_parser                                                               # The parameters in a job used in DASH-Sim are obtained from
                                                                                # Job initialization file (job_**.txt), parsed by job_parser.py
import processing_element                                                       # Define the processing element class
import DASH_Sim_core                                                            # The core of the simulation engine (SimulationManager) is defined DASH_Sim_core.py
import scheduler                                                                # The DASH-Sim uses the scheduler defined in scheduler.py
import DASH_Sim_utils
import DTPM_utils

def run_simulator(scale_values=common.scale_values_list):

    ## Open report file handles
    common.scale_values_list = common.str_to_list(scale_values)

    if common.ils_enable_dataset_save or common.ils_enable_dagger: #现在是no
        os.makedirs('./reports', exist_ok=True)
        common.open_report_file_handles(scale_values)
        common.scale_values_list = common.str_to_list(scale_values)

    if common.ils_enable_policy_decision or common.ils_enable_dataset_save :  #现在是no

        # Setup directories and files for IL-Scheduler
        if common.ils_enable_dataset_save or common.ils_enable_policy_decision or common.ils_enable_dagger :
            common.ils_setup(scale_values)
        ## if common.ils_enable_dataset_save or common.ils_enable_policy_decision or common.ils_enable_dagger :
    
        if common.ils_enable_dataset_save or common.ils_enable_dagger :
            common.ils_close_file_handles()
        ## if common.ils_enable_dataset_save or common.ils_enable_dagger :
                                                 # when user want to run the simulation on the command line
                                                                                    # type in command line python DASH_Sim_v0 -h

    plt.close('all')                                                                # close all existing plots before the new simulation
    if (common.CLEAN_TRACES):  #现在是yes
        DASH_Sim_utils.clean_traces()

    #common.clear_screen()                                                           # Clear IPthon Console screen at the beginning of each simulation
    print('%59s'%('**** Welcome to DASH_Sim.v0 ****'))
    print('%65s'%('**** \xa9 2018 eLab ASU ALL RIGHTS RESERVED ****'))


    # Instantiate the ResourceManager object that contains all the resources
    # in the target DSSoC
    #resource_matrix为ResourceManager对象，包括可用资源list[]和communication consumption[]
    resource_matrix = common.ResourceManager()                                      # This line generates an empty resource matrix
    config = configparser.ConfigParser()
    config.read('config_file.ini')
    resource_file = config['DEFAULT']['resource_file']
    #给resource_matrix.list和resource_matrix.comm_band赋值
    DASH_SoC_parser.resource_parse(resource_matrix, resource_file)    # Parse the input configuration file to populate the resource matrix
    # for i in resource_matrix.list:
    #     print('\n'.join(['%s:%s' % item for item in i.__dict__.items()]))
    #     print()
    # print('--------')



    for cluster in common.ClusterManager.cluster_list:
        if cluster.DVFS != 'none':
            if len(cluster.trip_freq) != len(common.trip_temperature) or len(cluster.trip_freq) != len(common.trip_hysteresis):
                print("[E] The trip points must match in size:")
                print("[E] Trip frequency (SoC file):      {} (Cluster {})".format(len(cluster.trip_freq), cluster.ID))
                print("[E] Trip temperature (config file): {}".format(len(common.trip_temperature)))
                print("[E] Trip hysteresis (config file):  {}".format(len(common.trip_hysteresis)))
                sys.exit()
            if len(cluster.power_profile) != len(cluster.PG_profile):
                print("[E] The power and PG profiles must match in size, please check the SoC file")
                print("[E] Cluster ID: {}, Num power points: {}, PG power points: {}".format(cluster.ID, len(cluster.power_profile), len(cluster.PG_profile)))
                sys.exit()

    # Instantiate the ApplicationManager object that contains all the jobs
    # in the target DSSoC
    jobs = common.ApplicationManager()                                              # This line generates an empty list for all jobs
    job_files_list = common.str_to_list(config['DEFAULT']['task_file'])
    for job_file in job_files_list:
        job_parser.job_parse(jobs, job_file)                                        # Parse the input job file to populate the job list

    # for i in range(len(jobs.list[1].comm_vol)):
    #     print('[',end="")
    #     for j in range(len(jobs.list[1].comm_vol[i])):
    #         print(jobs.list[1].comm_vol[i][j],end=" ")
    #     print(']')



    ## Initialize variables at simulation start
    DASH_Sim_utils.init_variables_at_sim_start()
    #common.max_num_jobs=10
    if len(common.job_list) > 0:
        common.max_num_jobs = int(config['SIMULATION MODE']['jobs']) * len(common.job_list)
    else:
        common.max_num_jobs = int(config['SIMULATION MODE']['jobs'])

    # 现在没有影响
    if not common.enable_real_time_constraints and common.enable_regression_policy:
        print("[E] The regression policy flag requires that the real-time constraint flag is enabled.")
        print("[E] Please enable the RT constraint flag, or disable the regression policy.")
        sys.exit()

    # 现在没有影响
    if common.enable_real_time_constraints:
        common.deadline_dict = DTPM_utils.get_snippet_deadlines()

    # Get the oracle configurations if using IL policy, otherwise, just load the hardware counter trace
    #现在没有影响
    if common.ClusterManager.cluster_list[0].DVFS == "imitation-learning":
        common.oracle_config_dict = DTPM_utils.get_oracle_frequencies_and_num_cores()
    else:
        #os.path.exists(common.HARDWARE_COUNTERS_TRACE)为false
        DTPM_utils.load_hardware_counters()

    # DTPM
    # 均为false，现在没有影响
    if os.path.exists(common.DTPM_freq_policy_file):
        common.IL_freq_policy = pickle.load(open(common.DTPM_freq_policy_file, 'rb'))
    else:
        common.IL_freq_policy = None

    if os.path.exists(common.DTPM_num_cores_policy_file):
        common.IL_num_cores_policy = pickle.load(open(common.DTPM_num_cores_policy_file, 'rb'))
    else:
        common.IL_num_cores_policy = None

    if os.path.exists(common.DTPM_regression_policy_file):
        common.IL_regression_policy = pickle.load(open(common.DTPM_regression_policy_file, 'rb'))
    else:
        common.IL_regression_policy = None

    # 选择调度方式
    scheduler_name = config['DEFAULT']['scheduler']                       # Assign the requested scheduler name to a variable

    # Check whether the resource_matrix and task list are initialized correctly
    if (common.DEBUG_CONFIG):
        print('\n[D] Starting DASH-Sim in DEBUG Mode ...')
        print("[D] Read the resource_matrix and write its contents")
        num_of_resources = len(resource_matrix.list)
        num_of_jobs = len(jobs.list)

        for i in range(num_of_resources):
            curr_resource = resource_matrix.list[i]
            print("[D] Adding a new resource: Type: %s, Name: %s, ID: %d, Capacity: %d" 
                  %(curr_resource.type, curr_resource.name, int(curr_resource.ID), int(curr_resource.capacity))) 
            print("[D] It supports the following %d functionalities"
                  %(curr_resource.num_of_functionalities))
    
            for ii in range(curr_resource.num_of_functionalities):
                print ('%4s'%('')+curr_resource.supported_functionalities[ii],
                       curr_resource.performance[ii])
        print('\nCommunication Bandwidth matrix between Resources is\n', common.ResourceManager.comm_band)
            # end for ii
        # end for i

        print("\n[D] Read each application and write its components")
        for ii in range(num_of_jobs):
            curr_job = jobs.list[ii]
            num_of_tasks = len(curr_job.task_list)
            print('\n%10s'%('')+'Now reading application %s' %(ii+1))
            print('Application name: %s, Number of tasks in the application: %s'%(curr_job.name, num_of_tasks))
            
            for task in jobs.list[ii].task_list:
                print("Task name: %s, Task ID: %s, Task Predecessor(s) %s"
                      %(task.name, task.ID, task.predecessors))
            print('Communication Volume matrix between Tasks is\n', jobs.list[ii].comm_vol)
        print(' ')
        # end for ii

        print('[D] Read the scheduler name')
        print('Scheduler name: %s' % scheduler_name)
        print('')
    # end if (DEBUG)

    # 现在是performance，现在无影响
    if (common.simulation_mode == 'validation'):
        '''
        Start the simulation in VALIDATION MODE
        '''
        job_execution_time = 0                                                  # Average execution time
        
        # Provide the value of the seed for the random variables
        random.seed(common.seed)  # user can regenerate the same results by assigning a value to $random_seed in configuration file
        np.random.seed(common.seed)
        common.iteration = 1 # set the iteration value

        # Instantiate the PerfStatics object that contains all the performance statics
        common.results = common.PerfStatics()

        # Set up the Python Simulation (simpy) environment
        env = simpy.Environment(initial_time=0)
        sim_done = env.event()

        # Construct the processing elements in the target DSSoC
        DASH_resources = []

        for i,resource in enumerate(resource_matrix.list):
            # Define the PEs (resources) in simpy environment
            new_PE = processing_element.PE(env, resource.type, resource.name,
                                           resource.ID, resource.cluster_ID, resource.capacity, resource.cost) # Generate a new PE with this generic process
            DASH_resources.append(new_PE)
        # end for

        # Construct the scheduler
        DASH_scheduler = scheduler.Scheduler(env, resource_matrix, scheduler_name,
                                             DASH_resources, jobs)

        # Check whether PEs are initialized correctly
        if (common.DEBUG_CONFIG):
            print('[D] There are %d simpy resources.' % len(DASH_resources))
            print('[D] Completed building and debugging the DASH model.\n')


        # Start the simulation engine
        print('[I] Starting the simulation under VALIDATION MODE...')

        job_gen = job_generator.JobGenerator(env, resource_matrix, jobs, DASH_scheduler, DASH_resources)

        sim_core = DASH_Sim_core.SimulationManager(env, sim_done, job_gen, DASH_scheduler, DASH_resources,
                                                  jobs, resource_matrix)


        env.run(until = common.simulation_length)
        
        job_execution_time += common.results.cumulative_exe_time / common.results.completed_jobs                        # find the mean job duration

        print('[I] Completed Simulation ...')
        for job in common.Validation.generated_jobs:
            if job in common.Validation.completed_jobs:
                continue
            else:
              print('[E] Not all generated jobs are completed')
              sys.exit()
        print('[I] And, simulation is validated, successfully.')
        print('\nSimulation Parameters')
        print("-"*55)
        print("%-30s : %-20s"%("SoC config file",resource_file))
        print("%-30s : %-20s"%("Job config files",' '.join(job_files_list)))
        print("%-30s : %-20s"%("Scheduler",scheduler_name))
        print("%-30s : %-20s"%("Clock period(us)",common.simulation_clk))
        print("%-30s : %-20d"%("Simulation length(us)",common.simulation_length))
        print('\nSimulation Statitics')
        print("-"*55)
        print("%-30s : %-20s" % ("Execution time(us)", round(common.results.execution_time, 2)))
        print("%-30s : %-20s"%("Avg execution time(us)",job_execution_time))
        print("%-30s : %-20s" % ("Total energy consumption(uJ)",
                                 round(common.results.energy_consumption, 2)))
        print("%-30s : %-20s" % ("EDP",
                                 round(common.results.execution_time * common.results.energy_consumption, 2)))
        DASH_Sim_utils.trace_system()
        # End of simpy simulation

        plot_gantt_chart = True
        if plot_gantt_chart:
            # Creating a text based Gantt chart to visualize the simulation
            job_ID = -1
            ilen = len(resource_matrix.list) - 1  # since the last PE is the memory
            pos = np.arange(0.5, ilen * 0.5 + 0.5, 0.5)
            fig = plt.figure(figsize=(10, 4.5))
            # fig = plt.figure(figsize=(10,3.5))
            ax = fig.add_subplot(111)
            # color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']
            # color_choices = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:pink']
            color_choices = plt.get_cmap('Set2').colors
            color_choices = color_choices[0::int(len(color_choices)/5)]
            # color_choices = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html#matplotlib.patches.Patch.set_hatch
            # hatch_choices = [None, '/', '\\', 'x', '.', '+', '*']
            hatch_choices = [None, '////', '\\\\\\\\', '....', 'x', '.', '+', '*']
            for i in range(len(resource_matrix.list)):
                for ii, task in enumerate(common.TaskQueues.completed.list):
                    if (i == task.PE_ID):
                        end_time = task.finish_time
                        start_time = task.start_time
                        # ax.barh((i * 0.5) + 0.5, end_time - start_time, left=start_time,
                        #         height=0.3, align='center', edgecolor='black', color='white', alpha=0.95)
                        ax.barh((i * 0.5) + 0.5, end_time - start_time, left=start_time,
                                height=0.3, align='center', edgecolor='black', color=color_choices[(task.jobID) % 5], alpha=0.95, 
                                hatch=hatch_choices[(task.jobID) % 5])
                        # Retrieve the job ID which the current task belongs to
                        for iii, job in enumerate(jobs.list):
                            if (job.name == task.jobname):
                                job_ID = iii
                        #ax.text(0.5 * (start_time + end_time - len(str(task.ID)) - 0.25), (i * 0.5) + 0.5 - 0.03125,
                        #        task.ID, color=color_choices[(task.jobID) % 5], fontweight='bold', fontsize=18, alpha=0.75)
            # color_choices[(task.jobID)% 5]
            # color_choices[job_ID]
            # locsy, labelsy = plt.yticks(pos, ['P0','P1','P2']) #
            locsy, labelsy = plt.yticks(pos, range(len(resource_matrix.list)))
            plt.ylabel('Processing Element', fontsize=18)
            plt.xlabel('Time', fontsize=18)
            plt.tick_params(labelsize=16)
            # plt.title('DASH-Sim - %s' %(scheduler_name), fontsize =18)
            plt.setp(labelsy, fontsize=18)
            ax.set_ylim(bottom=-0.1, top=ilen * 0.5 + 0.5)
            ax.set_xlim(left=-5, right=1020)
            ax.grid(color='g', linestyle=':', alpha=0.5)
            plt.axvline(995, color='black', linestyle='--')
            ax.text(995 - 90, 5, "995", fontweight='bold', fontsize=18)
            #plt.show()
            plt.savefig("ds3_gantt.pdf", bbox_inches='tight')#, dpi=300)
    # end of if (common.simulation_mode == 'validation'):

    if (common.simulation_mode == 'performance'):
        '''
        Start the simulation in PERFORMANCE MODE
        '''
        #len(common.scale_values_list)=1
        ave_job_injection_rate = [0]*len(common.scale_values_list)                  # The list contains the mean of the lambda injection value corresponding each lambda value 
                                                                                    # Based on the number of jobs put into ready queue list
        ave_job_execution_time = [0]*len(common.scale_values_list)                  # The list contains the mean job duration for each lambda value
        ave_job_completion_rate = [0]*len(common.scale_values_list)                 # The list contains the mean job completion rate for each lambda value
        lamd_values_list = [0]*len(common.scale_values_list)                        # The list of lambda values which will determine the job arrival rate
        ave_concurrent_jobs = [0]*len(common.scale_values_list)                     # Average number of jobs in the system for a workload with a specific scale value

        #ind=0 scale=2000
        for (ind, scale) in enumerate(common.scale_values_list):

            common.scale = scale  # Assign each value in $scale_values_list to common.scale
            lamd_values_list[ind] = 1 / scale

            #yes
            if (common.INFO_JOB):
                print('[I] Simulation starts for scale value %s' %(scale))

            # Iterate over a fixed number of iterations
            job_execution_time  = 0.0
            job_injection_rate  = 0.0
            job_completion_rate = 0.0
            concurrent_jobs     = 0.0
            # common.num_of_iterations=3
            for iteration in range(common.num_of_iterations):                       # Repeat the simulation for a given number of numbers for each lambda value
                # Initialize variables at simulation start
                DASH_Sim_utils.init_variables_at_sim_start()

                # Set a global iteration variable
                common.iteration = iteration

                # 随机种子相同时产生相同的随机数
                random.seed(iteration)     # user can regenerate the same results by assigning a value to $random_seed in configuration file
                np.random.seed(iteration)

                # Instantiate the PerfStatics object that contains all the performance statics
                common.results = common.PerfStatics()
                common.computation_dict = {}
                common.current_dag = nx.DiGraph()  #digraph有向带权图

                # Set up the Python Simulation (simpy) environment
                env = simpy.Environment(initial_time=0)
                sim_done = env.event()
                # Construct the processing elements in the target DSSoC
                DASH_resources = []
                for i,resource in enumerate(resource_matrix.list):
                    # Define the PEs (resources) in simpy environment
                    new_PE = processing_element.PE(env, resource.type, resource.name,
                                                   resource.ID, resource.cluster_ID, resource.capacity, resource.cost) # Generate a new PE with this generic process
                    DASH_resources.append(new_PE)
                # end for


                # Construct the scheduler
                DASH_scheduler = scheduler.Scheduler(env, resource_matrix, scheduler_name,
                                                     DASH_resources, jobs)
                if (common.INFO_JOB):
                    print('[I] Starting iteration: %d' %(iteration+1))

                # HEFT/PEFT在其中调用
                # 生成outstanding_list和ready_list
                job_gen = job_generator.JobGenerator(env, resource_matrix, jobs, DASH_scheduler, DASH_resources)

                # 调用调度算法
                #生成running_list
                sim_core = DASH_Sim_core.SimulationManager(env, sim_done, job_gen, DASH_scheduler, DASH_resources,
                                                           jobs, resource_matrix)

                # for i in DASH_resources:
                #     print('\n'.join(['%s:%s' % item for item in i.__dict__.items()]))
                #     print()
                # print('--------')
                if common.sim_early_stop is False:
                    env.run(until=common.simulation_length) #运行时长100000个时间单位
                else:
                    env.run(until=sim_done)

                # Now, the simulation has completed
                # Next, process the results
                # 进入该分支
                if (common.INFO_JOB):
                    # print('[I] Completed iteration: %d' %(iteration+1))
                    # print('[I] Number of injected jobs: %d' %(common.results.injected_jobs))
                    # print('[I] Number of completed jobs: %d' %(common.results.completed_jobs))
                    # print('[I] Number of jobs: %d' %(common.num_of_jobs_1))
                    # print('[I] Number of failed jobs: %d' %(common.num_of_out_1))
                    # try:
                    #     print('[I] Ave latency: %f'
                    #     %(common.results.cumulative_exe_time/common.results.completed_jobs))
                    # except ZeroDivisionError:
                    #     print('[I] No completed jobs')
                    # print("[I] %-30s : %-20s" % ("Execution time(us)", round(common.results.execution_time - common.warmup_period, 2)))
                    # print("[I] %-30s : %-20s" % ("Cumulative Execution time(us)", round(common.results.cumulative_exe_time, 2)))
                    # print("[I] %-30s : %-20s" % ("Total energy consumption(J)",
                    #                              round(common.results.cumulative_energy_consumption, 6)))
                    # print("[I] %-30s : %-20s" % ("EDP",
                    #                              round((common.results.execution_time - common.warmup_period) * common.results.cumulative_energy_consumption, 2)))
                    # print("[I] %-30s : %-20s" % ("Cumulative EDP",
                    #                              round(common.results.cumulative_exe_time * common.results.cumulative_energy_consumption, 2)))
                    # # print("[I] %-30s : %-20s" % ("Cumulative EDDP",
                    # #                              round(common.results.cumulative_exe_time * common.results.cumulative_exe_time * common.results.cumulative_energy_consumption, 2)))
                    # print("[I] %-30s : %-20s" % ("Average concurrent jobs", round(common.results.average_job_number, 2)))
                    injection_rate = common.results.injected_jobs / (common.simulation_length - common.warmup_period)
                    result_exec_time = common.results.execution_time - common.warmup_period
                    cumulative_exec_time = common.results.cumulative_exe_time
                    result_energy_cons = common.results.cumulative_energy_consumption
                    result_EDP = result_exec_time * result_energy_cons
                    cumulative_EDP = cumulative_exec_time * result_energy_cons
                    jobs_completed = common.results.completed_jobs
                    header_list = ['Injection Rate', 'Execution time(us)', 'Cumulative execution time(us)', 'Total energy consumption(J)', 'EDP', 'Cumulative EDP', 'Completed Jobs', 'Scale Value', 'Iteration', 'Scheduler']
                    result_list = [injection_rate, result_exec_time, cumulative_exec_time, result_energy_cons, result_EDP, cumulative_EDP, jobs_completed, scale, iteration, scheduler_name]
                    if common.total_predictions > 0:
                        print("%-30s : %-20s" % ("Total predictions",
                                                 common.total_predictions))
                        print("%-30s : %-20s" % ("Wrong predictions (freq)",
                                                 common.wrong_predictions_freq))
                        header_list.append('Total predictions')
                        result_list.append(common.total_predictions)
                        header_list.append('Wrong predictions (freq)')
                        result_list.append(common.wrong_predictions_freq)
                        if common.enable_num_cores_prediction:
                            print("%-30s : %-20s" % ("Wrong predictions (num_cores)",
                                                     common.wrong_predictions_num_cores))
                            header_list.append('Wrong predictions (num_cores)')
                            result_list.append(common.wrong_predictions_num_cores)
                        print("%-30s : %-20s" % ("Accuracy (freq)",
                                                 ((common.total_predictions - common.wrong_predictions_freq) / common.total_predictions) * 100))
                        header_list.append('Accuracy (freq)')
                        result_list.append(((common.total_predictions - common.wrong_predictions_freq) / common.total_predictions) * 100)
                        if common.enable_num_cores_prediction:
                            print("%-30s : %-20s" % ("Accuracy (num_cores)",
                                                     ((common.total_predictions - common.wrong_predictions_num_cores) / common.total_predictions) * 100))
                            header_list.append('Accuracy (num_cores)')
                            result_list.append(((common.total_predictions - common.wrong_predictions_num_cores) / common.total_predictions) * 100)
                    if common.enable_real_time_constraints and common.snippet_ID_exec > 0:
                        print("%-30s : %-20s" % ("Missed deadlines:",
                                                 "{} ({:.2f}%)".format(common.missed_deadlines, (common.missed_deadlines / (common.snippet_ID_exec)) * 100)))
                        header_list.append('Missed deadlines')
                        result_list.append(common.missed_deadlines)
                        header_list.append('Missed deadlines (%)')
                        result_list.append((common.missed_deadlines / (common.snippet_ID_exec)) * 100)
                    DASH_Sim_utils.trace_system()
                    if not common.generate_complete_trace:
                        if not os.path.exists(common.RESULTS):
                            with open(common.RESULTS, 'w', newline='') as csvfile:
                                result_file = csv.writer(csvfile, delimiter=',')
                                result_file.writerow(header_list)
                        with open(common.RESULTS, 'a', newline='') as csvfile:
                            result_file = csv.writer(csvfile, delimiter=',')
                            result_file.writerow(result_list)
                        if common.ClusterManager.cluster_list[0].DVFS == "imitation-learning":
                            common.oracle_config_dict = DTPM_utils.get_oracle_frequencies_and_num_cores()
                            DTPM_utils.update_oracle_dataset('freq', 'complete')
                            DTPM_utils.update_oracle_dataset('freq', 'reduced')
                            if common.enable_num_cores_prediction:
                                DTPM_utils.update_oracle_dataset('num_cores', 'complete')
                                DTPM_utils.update_oracle_dataset('num_cores', 'reduced')
                            if common.enable_real_time_constraints:
                                DTPM_utils.update_oracle_dataset('regression', 'complete')
                                DTPM_utils.update_oracle_dataset('regression', 'reduced')

                try:
                    job_execution_time += common.results.cumulative_exe_time / common.results.completed_jobs                    # find the mean job duration value for this iteration
                except ZeroDivisionError:
                    job_execution_time += 0

                job_injection_rate += common.results.injected_jobs * 1000 / (common.time_at_sim_termination - common.warmup_period)      # find the average injection rate
                job_completion_rate += common.results.completed_jobs * 1000 / (common.time_at_sim_termination - common.warmup_period)    # find the average injection rate

                # 不进入该分支
                if common.ils_enable_dataset_save or common.ils_enable_dagger:
                    common.report_fp.write('[I] Completed iteration: %d\n' %(iteration+1))
                    common.report_fp.write('[I] Number of injected jobs: %d\n' %(common.results.injected_jobs))
                    common.report_fp.write('[I] Number of completed jobs: %d\n' %(common.results.completed_jobs))
                    try:
                        common.report_fp.write('[I] Ave latency: %f\n'
                        %(common.results.cumulative_exe_time/common.results.completed_jobs))
                    except ZeroDivisionError:
                        common.report_fp.write('[I] No completed jobs\n')
                    common.report_fp.write("[I] %-30s : %-20s\n" % ("Average concurrent jobs", round(common.results.average_job_number, 2)))
                    common.report_fp.write("[I] %-30s : %-20s\n" % ("Execution time(us)", round(common.results.execution_time - common.warmup_period, 2)))
                    common.report_fp.write("[I] %-30s : %-20s\n" % ("Cumulative Execution time(us)", round(common.results.cumulative_exe_time, 2)))

                job_injection_rate += common.results.injected_jobs / (common.simulation_length - common.warmup_period)      # find the average injection rate
                job_completion_rate += common.results.completed_jobs / (common.simulation_length - common.warmup_period)    # find the average injection rate
                concurrent_jobs += common.results.average_job_number
                print()
            # end of for iteration in range(common.num_of_iterations):

            ave_job_execution_time[ind] = job_execution_time / common.num_of_iterations
            ave_job_injection_rate[ind] = job_injection_rate / common.num_of_iterations
            ave_job_completion_rate[ind] = job_completion_rate / common.num_of_iterations
            ave_concurrent_jobs[ind] = concurrent_jobs / common.num_of_iterations

            if (common.INFO_JOB):
                print('[I] Completed all %d iterations for scale = %d, number of injection jobs = %d, number of overtime jobs = %d'
                      %(common.num_of_iterations, scale, common.num_of_jobs, common.num_of_out), end=' ')
                print('ave_rate_of_out_deadline:%f' % (common.num_of_out/common.num_of_jobs))
                print('Job Statistics')
                print(sorted(common.job_statistics.items(), key=lambda x: x[0]))
                print('Overtime Job Statistics:')
                print(sorted(common.overtime_job_statistics.items(), key=lambda x: x[0]))
                print('Overtime Statistics:')
                print(sorted(common.overtime_job_time_statistics.items(), key=lambda x: x[0]))
                sum = 0
                for i in common.overtime_job_time_statistics.items():
                    sum += i[1]
                print("Total Delay:", sum)
                print("Average Delay:", float(sum/1000))
                # print("exe_time:", float(common.exe_time / common.task_num * 100))
                print("exe_time:", float(common.exe_time / common.simulation_num))
                print(common.times)
                # print(' injection rate:%f, completion rate:%f, ave_execution_time:%f, ave_num_of_outdeadline:%f'
                #       % (ave_job_injection_rate[ind], ave_job_completion_rate[ind], ave_job_execution_time[ind], common.num_of_out/common.num_of_jobs))

            if common.ils_enable_dataset_save or common.ils_enable_dagger:
                common.report_fp.write('[I] Completed all %d iterations for scale = %d,'
                      %(common.num_of_iterations,scale))
                common.report_fp.write(' injection rate:%f, concurrent_jobs:%f, completion rate:%f, ave_execution_time:%f\n'
                      % (ave_job_injection_rate[ind], ave_concurrent_jobs[ind], ave_job_completion_rate[ind], ave_job_execution_time[ind]))
                common.report_fp.close()


        # end of for (ind,scale) in enumerate(common.scale_values_list):


        # fieldnames = ['injection_rate', 'execution','conc. jobs']
        # rows = zip(ave_job_injection_rate,ave_job_execution_time,ave_concurrent_jobs)
        # with open('output.csv', 'w') as f:
        #     writer = csv.DictWriter(f, lineterminator='\n', fieldnames = fieldnames)
        #     writer.writeheader()
        #     writer = csv.writer(f, lineterminator='\n', )
        #     for row in rows:
        #         writer.writerow(row)

if __name__ == '__main__':
    run_simulator(common.config_scale_values)
