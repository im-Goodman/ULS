'''
Descrption: This file contains the code for scheduler class 
which contains different types of scheduler. 
'''
import time
from collections import deque
from math import inf
from types import SimpleNamespace

import networkx as nx
import numpy as np
import copy

import common  # The common parameters used in DASH-Sim are defined in common_parameters.py
from ULS.ULS import getTask_10
from heft import heft, dag_merge
from heftrt import heft as heftrt
from peft import peft, gantt
import DTPM_power_models


class Scheduler:
    def __init__(self, env, resource_matrix, name, PE_list, jobs):
        '''
        env: Pointer to the current simulation environment
        resource_matrix: The data structure that defines power/performance
    		  characteristics of the PEs for each supported task
        name : The name of the requested scheduler
        PE_list: The PEs available in the current SoCs
        jobs: The list of all jobs given to DASH-Sim
        '''
        self.env = env
        self.resource_matrix = resource_matrix
        self.name = name
        self.PEs = PE_list
        self.jobs = jobs
        self.assigned = [0] * (len(self.PEs))

        # At the end of this function, the scheduler class has a copy of the
        # the power/performance characteristics of the resource matrix and
        # name of the requested scheduler name

    # end  def __init__(self, env, resource_matrix, scheduler_name)

    # Specific scheduler instances can be defined below
    def CPU_only(self, list_of_ready):
        '''
        This scheduler always select the resource with ID 0 (CPU) to
        execute all outstanding tasks without any comparison between
        available resources
        '''
        for task in list_of_ready:
            task.PE_ID = 0

    # end def CPU_only(list_of_ready):

    def MET(self, list_of_ready):
        '''
        This scheduler compares the execution times of the current
        task for available resources and returns the ID of the resource
        with minimum execution time for the current task.
        '''
        if common.deadline_type == 'mix':
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 144
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 903
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 162
            elif common.job_name_temp == 'Top':
                deadline = 93
            elif common.job_name_temp == 'lag_detection':
                deadline = 531
        elif common.deadline_type == 'tight':
            # np.random.seed(common.iteration)
            s = common.s
            if s == 0:
                if common.job_name_temp == 'WiFi_Transmitter':
                    deadline = 216
                elif common.job_name_temp == 'WiFi_Receiver':
                    deadline = 903
                elif common.job_name_temp == 'Temporal_Mitigation':
                    deadline = 243
                elif common.job_name_temp == 'Top':
                    deadline = 93
                elif common.job_name_temp == 'lag_detection':
                    deadline = 531
            else:
                if common.job_name_temp == 'WiFi_Transmitter':
                    deadline = 144
                elif common.job_name_temp == 'WiFi_Receiver':
                    deadline = 602
                elif common.job_name_temp == 'Temporal_Mitigation':
                    deadline = 162
                elif common.job_name_temp == 'Top':
                    deadline = 62
                elif common.job_name_temp == 'lag_detection':
                    deadline = 354
        else:
            # np.random.seed(common.iteration)
            s = common.s
            if s == 0:
                if common.job_name_temp == 'WiFi_Transmitter':
                    deadline = 216
                elif common.job_name_temp == 'WiFi_Receiver':
                    deadline = 903
                elif common.job_name_temp == 'Temporal_Mitigation':
                    deadline = 243
                elif common.job_name_temp == 'Top':
                    deadline = 93
                elif common.job_name_temp == 'lag_detection':
                    deadline = 531
            else:
                if common.job_name_temp == 'WiFi_Transmitter':
                    deadline = 144
                elif common.job_name_temp == 'WiFi_Receiver':
                    deadline = 602
                elif common.job_name_temp == 'Temporal_Mitigation':
                    deadline = 162
                elif common.job_name_temp == 'Top':
                    deadline = 62
                elif common.job_name_temp == 'lag_detection':
                    deadline = 354
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = self.env.now + deadline
        # Initialize a list to record number of assigned tasks to a PE
        # for every scheduling instance
        assigned = [0] * (len(self.PEs))

        # go over all ready tasks for scheduling and make a decision
        for task in list_of_ready:
            # print(task.ID)
            exec_times = [np.inf] * (len(self.PEs))  # Initialize a list to keep execution times of task for each PE

            for i in range(len(self.resource_matrix.list)):
                if self.PEs[i].enabled:
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):
                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
                        exec_times[i] = self.resource_matrix.list[i].performance[ind]
            # print(exec_times)
            min_of_exec_times = min(
                exec_times)  # $min_of_exec_times is the minimum of execution time of the task among all PEs
            count_minimum = exec_times.count(
                min_of_exec_times)  # also, record how many times $min_of_exec_times is seen in the list
            # print(count_minimum)

            # if there are two or more PEs satisfying minimum execution
            # then we should try to utilize all those PEs
            if (count_minimum > 1):

                # if there are tow or more PEs satisfying minimum execution
                # populate the IDs of those PEs into a list
                min_PE_IDs = [i for i, x in enumerate(exec_times) if x == min_of_exec_times]

                # then check whether those PEs are busy or idle
                PE_check_list = [True if not self.PEs[index].idle else False for i, index in enumerate(min_PE_IDs)]

                # assign tasks to the idle PEs instead of the ones that are currently busy
                if (True in PE_check_list) and (False in PE_check_list):
                    for PE in PE_check_list:
                        # if a PE is currently busy remove that PE from $min_PE_IDs list
                        # to schedule the task to a idle PE
                        if (PE == True):
                            min_PE_IDs.remove(min_PE_IDs[PE_check_list.index(PE)])

                # then compare the number of the assigned tasks to remaining PEs
                # and choose the one with the lowest number of assigned tasks
                assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
                PE_ID_index = assigned_tasks.index(min(assigned_tasks))

                # finally, choose the best available PE for the task
                task.PE_ID = min_PE_IDs[PE_ID_index]

                # =============================================================================
                #                 # assign tasks to the idle PEs instead of the ones that are currently busy
                #                 if (True in PE_check_list) and (False in PE_check_list):
                #                     for PE in PE_check_list:
                #                         # if a PE is currently busy remove that PE from $min_PE_IDs list
                #                         # to schedule the task to a idle PE
                #                         if (PE == True):
                #                             min_PE_IDs.remove(min_PE_IDs[PE_check_list.index(PE)])
                #
                #
                #                 # then compare the number of the assigned tasks to remaining PEs
                #                 # and choose the one with the lowest number of assigned tasks
                #                 assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
                #                 PE_ID_index = assigned_tasks.index(min(assigned_tasks))
                # =============================================================================

                # finally, choose the best available PE for the task
                task.PE_ID = min_PE_IDs[PE_ID_index]

            else:
                task.PE_ID = exec_times.index(min_of_exec_times)
            # end of if count_minimum >1:
            # since one task is just assigned to a PE, increase the number by 1
            assigned[task.PE_ID] += 1

            if (task.PE_ID == -1):
                print('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                      % (self.env.now, task.name))
                print('[E] or job_**.txt file')
                assert (task.PE_ID >= 0)
            else:
                if (common.INFO_SCH):
                    print('[I] Time %s: The scheduler assigns the %s task to resource PE-%s: %s'
                          % (self.env.now, task.ID, task.PE_ID,
                             self.resource_matrix.list[task.PE_ID].type))
            # end of if task.PE_ID == -1:
        # end of for task in list_of_ready:
        # At the end of this loop, we should have a valid (non-negative ID)
        # that can run next_task

    # end of MET(list_of_ready)

    def EFT(self, list_of_ready):
        '''
        This scheduler compares the execution times of the current
        task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the
        earliest finish time for the task
        '''
        if not list_of_ready:
            return
        s = common.s
        if s == 0:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 213
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 873
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 237
            elif common.job_name_temp == 'Top':
                deadline = 93
            elif common.job_name_temp == 'lag_detection':
                deadline = 531
        else:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 142
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 582
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 158
            elif common.job_name_temp == 'Top':
                deadline = 62
            elif common.job_name_temp == 'lag_detection':
                deadline = 354
        if common.job_name_temp == 'TEST':
            deadline = 35
        elif common.job_name_temp == 'TEST1':
            deadline = 48
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = common.arrive_time[task.jobID] + deadline

        for task in list_of_ready:

            comparison = [np.inf] * len(self.PEs)  # Initialize the comparison vector
            comm_ready = [0] * len(self.PEs)  # A list to store the max communication times for each PE

            if (common.DEBUG_SCH):
                print('[D] Time %s: The scheduler function is called with task %s'
                      % (self.env.now, task.ID))

            for i in range(len(self.resource_matrix.list)):
                if self.PEs[i].enabled:
                    # if the task is supported by the resource, retrieve the index of the task
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):
                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)

                        # $PE_comm_wait_times is a list to store the estimated communication time
                        # (or the remaining communication time) of all predecessors of a task for a PE
                        # As simulation forwards, relevant data is being sent after a task is completed
                        # based on the time instance, one should consider either whole communication
                        # time or the remaining communication time for scheduling
                        PE_comm_wait_times = []

                        # $PE_wait_time is a list to store the estimated wait times for a PE
                        # till that PE is available if the PE is currently running a task
                        PE_wait_time = []

                        job_ID = -1  # Initialize the job ID

                        # Retrieve the job ID which the current task belongs to
                        for ii, job in enumerate(self.jobs.list):
                            if job.name == task.jobname:
                                job_ID = ii

                        for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                            # data required from the predecessor for $ready_task
                            c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                            # retrieve the real ID  of the predecessor based on the job ID
                            real_predecessor_ID = predecessor + task.ID - task.base_ID

                            # Initialize following two variables which will be used if
                            # PE to PE communication is utilized
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1

                            for completed in common.TaskQueues.completed.list:
                                if completed.ID == real_predecessor_ID:
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                                    # print(predecessor, predecessor_finish_time, predecessor_PE_ID)

                            if (common.PE_to_PE):
                                # Compute the PE to PE communication time
                                PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                                PE_comm_wait_times.append(
                                    max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))

                                if (common.DEBUG_SCH):
                                    print(
                                        '[D] Time %s: Estimated communication time between PE %s to PE %s from task %s to task %s is %d'
                                        % (self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID,
                                           PE_comm_wait_times[-1]))

                            if (common.shared_memory):
                                # Compute the communication time considering the shared memory
                                # only consider memory to PE communication time
                                # since the task passed the 1st phase (PE to memory communication)
                                # and its status changed to ready

                                # PE_to_memory_band = common.ResourceManager.comm_band[predecessor_PE_ID, -1]
                                memory_to_PE_band = common.ResourceManager.comm_band[
                                    self.resource_matrix.list[-1].ID, i]
                                shared_memory_comm_time = int(c_vol / memory_to_PE_band)

                                PE_comm_wait_times.append(shared_memory_comm_time)
                                if (common.DEBUG_SCH):
                                    print(
                                        '[D] Time %s: Estimated communication time between memory to PE %s from task %s to task %s is %d'
                                        % (self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                            # $comm_ready contains the estimated communication time
                            # for the resource in consideration for scheduling
                            # maximum value is chosen since it represents the time required for all
                            # data becomes available for the resource.
                            comm_ready[i] = (max(PE_comm_wait_times))
                        # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                        # if a resource currently is executing a task, then the estimated remaining time
                        # for the task completion should be considered during scheduling
                        PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                        # update the comparison vector accordingly
                        comparison[i] = self.resource_matrix.list[i].performance[ind] + max(comm_ready[i],
                                                                                            PE_wait_time[-1])
                    # end of if (task.name in...
            # end of for i in range(len(self.resource_matrix.list)):

            # after going over each resource, choose the one which gives the minimum result
            task.PE_ID = comparison.index(min(comparison))

            if task.PE_ID == -1:
                print('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                      % (self.env.now, task.ID))
                print('[E] or job_**.txt file')
                assert (task.PE_ID >= 0)
            else:
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively'
                          % (self.env.now, task.ID))
                    print('%12s' % (''), comparison)
                    print('[D] Time %s: The scheduler assigns task %s to resource %s: %s'
                          % (self.env.now, task.ID, task.PE_ID, self.resource_matrix.list[task.PE_ID].type))

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            self.PEs[task.PE_ID].available_time = self.env.now + comparison[task.PE_ID]

            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of for task in list_of_ready:

    # end of EFT(list_of_ready)

    def STF(self, list_of_ready):
        '''
        This scheduler compares the execution times of the current
        task for available resources and returns the ID of the resource
        with minimum execution time for the current task. The only difference
        between STF and MET is the order in which the tasks are scheduled onto resources
        '''

        ready_list = copy.deepcopy(list_of_ready)
        # Iterate through the list of ready tasks until all of them are scheduled
        while (len(ready_list) > 0):

            shortest_task_exec_time = np.inf
            shortest_task_pe_id = -1

            for task in ready_list:

                min_time = np.inf  # Initialize the best performance found so far as a large number

                for i in range(len(self.resource_matrix.list)):
                    if self.PEs[i].enabled:
                        if (task.name in self.resource_matrix.list[i].supported_functionalities):
                            ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)

                            if (self.resource_matrix.list[i].performance[
                                ind] < min_time):  # Found resource with smaller execution time
                                min_time = self.resource_matrix.list[i].performance[
                                    ind]  # Update the best time found so far
                                resource_id = self.resource_matrix.list[i].ID  # Record the ID of the resource
                                # task.PE_ID = i                                                          # Record the corresponding resource

                # print('[INFO] Task - %d, Resource - %d, Time - %d' %(task.ID, resource_id, min_time))
                # Obtain the ID and resource for the shortest task in the current iteration
                if (min_time < shortest_task_exec_time):
                    shortest_task_exec_time = min_time
                    shortest_task_pe_id = resource_id
                    shortest_task = task
                # end of if (min_time < shortest_task_exec_time)

            # end of for task in list_of_ready:
            # At the end of this loop, we should have the minimum execution time
            # of a task across all resources

            # Assign PE ID of the shortest task
            index = [i for i, x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            shortest_task.PE_ID = shortest_task_pe_id

            if (common.DEBUG_SCH):
                print('[I] Time %s: The scheduler function found task %d to be shortest on resource %d with %.1f'
                      % (self.env.now, shortest_task.ID, shortest_task.PE_ID, shortest_task_exec_time))

            if list_of_ready[index].PE_ID == -1:
                print('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                      % (self.env.now, shortest_task.name))
                print('[E] or job_**.txt file')
                assert (shortest_task.PE_ID >= 0)
            else:
                if (common.INFO_SCH):
                    print('[I] Time %s: The scheduler assigns the %s task to resource PE-%s: %s'
                          % (self.env.now, shortest_task.ID, shortest_task.PE_ID,
                             self.resource_matrix.list[shortest_task.PE_ID].type))
            # end of if shortest_task.PE_ID == -1:

            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list):
                if task.ID == shortest_task.ID:
                    ready_list.remove(task)

        # end of for task in list_of_ready:
        # At the end of this loop, all ready tasks are assigned to the resources
        # on which the execution times are minimum. The tasks will execute
        # in the order of increasing execution times

    # end of STF(list_of_ready)

    def ETF_LB(self, list_of_ready):
        '''
        This scheduler compares the execution times of the current
        task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the
        earliest finish time for the task. Additionally, the task with the
        lowest earliest finish time  is scheduled first
        '''
        s = common.s
        if s == 0:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 213
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 873
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 237
            elif common.job_name_temp == 'Top':
                deadline = 93
            elif common.job_name_temp == 'lag_detection':
                deadline = 531
        else:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 142
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 582
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 158
            elif common.job_name_temp == 'Top':
                deadline = 62
            elif common.job_name_temp == 'lag_detection':
                deadline = 354
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = common.arrive_time[task.jobID] + deadline
        ready_list = copy.deepcopy(list_of_ready)

        task_counter = 0
        assigned = self.assigned

        # Iterate through the list of ready tasks until all of them are scheduled
        while len(ready_list) > 0:

            shortest_task_exec_time = np.inf
            shortest_task_pe_id = -1
            shortest_comparison = [np.inf] * len(self.PEs)

            for task in ready_list:

                comparison = [np.inf] * len(self.PEs)  # Initialize the comparison vector
                comm_ready = [0] * len(self.PEs)  # A list to store the max communication times for each PE

                if (common.DEBUG_SCH):
                    print('[D] Time %s: The scheduler function is called with task %s'
                          % (self.env.now, task.ID))

                for i in range(len(self.resource_matrix.list)):
                    if self.PEs[i].enabled:
                        # if the task is supported by the resource, retrieve the index of the task
                        if (task.name in self.resource_matrix.list[i].supported_functionalities):
                            ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)

                            # $PE_comm_wait_times is a list to store the estimated communication time
                            # (or the remaining communication time) of all predecessors of a task for a PE
                            # As simulation forwards, relevant data is being sent after a task is completed
                            # based on the time instance, one should consider either whole communication
                            # time or the remaining communication time for scheduling
                            PE_comm_wait_times = []

                            # $PE_wait_time is a list to store the estimated wait times for a PE
                            # till that PE is available if the PE is currently running a task
                            PE_wait_time = []

                            job_ID = -1  # Initialize the job ID

                            # Retrieve the job ID which the current task belongs to
                            for ii, job in enumerate(self.jobs.list):
                                if job.name == task.jobname:
                                    job_ID = ii

                            for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                                # data required from the predecessor for $ready_task
                                c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                                # retrieve the real ID  of the predecessor based on the job ID
                                real_predecessor_ID = predecessor + task.ID - task.base_ID

                                # Initialize following two variables which will be used if
                                # PE to PE communication is utilized
                                predecessor_PE_ID = -1
                                predecessor_finish_time = -1

                                for completed in common.TaskQueues.completed.list:
                                    if completed.ID == real_predecessor_ID:
                                        predecessor_PE_ID = completed.PE_ID
                                        predecessor_finish_time = completed.finish_time
                                        # print(predecessor, predecessor_finish_time, predecessor_PE_ID)
                                        break

                                if (common.PE_to_PE):
                                    # Compute the PE to PE communication time
                                    PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                    PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                                    PE_comm_wait_times.append(
                                        max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))

                                    if (common.DEBUG_SCH):
                                        print(
                                            '[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d'
                                            % (self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID,
                                               PE_comm_wait_times[-1]))

                                if (common.shared_memory):
                                    # Compute the communication time considering the shared memory
                                    # only consider memory to PE communication time
                                    # since the task passed the 1st phase (PE to memory communication)
                                    # and its status changed to ready

                                    # PE_to_memory_band = common.ResourceManager.comm_band[predecessor_PE_ID, -1]
                                    memory_to_PE_band = common.ResourceManager.comm_band[
                                        self.resource_matrix.list[-1].ID, i]
                                    shared_memory_comm_time = int(c_vol / memory_to_PE_band)

                                    PE_comm_wait_times.append(shared_memory_comm_time)
                                    if (common.DEBUG_SCH):
                                        print(
                                            '[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d'
                                            % (self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))

                                # $comm_ready contains the estimated communication time
                                # for the resource in consideration for scheduling
                                # maximum value is chosen since it represents the time required for all
                                # data becomes available for the resource.
                                comm_ready[i] = max(PE_comm_wait_times)
                            # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                            # if a resource currently is executing a task, then the estimated remaining time
                            # for the task completion should be considered during scheduling
                            PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                            # update the comparison vector accordingly
                            comparison[i] = self.resource_matrix.list[i].performance[ind] * (
                                    1 + DTPM_power_models.compute_DVFS_performance_slowdown(
                                common.ClusterManager.cluster_list[self.PEs[i].cluster_ID])) + max(comm_ready[i],
                                                                                                   PE_wait_time[-1])
                        # end of if (task.name in...
                # end of for i in range(len(self.resource_matrix.list)):

                if min(comparison) < shortest_task_exec_time:
                    resource_id = comparison.index(min(comparison))
                    shortest_task_exec_time = min(comparison)
                    #                    print(shortest_task_exec_time, comparison)
                    count_minimum = comparison.count(
                        shortest_task_exec_time)  # also, record how many times $min_of_exec_times is seen in the list
                    # if there are two or more PEs satisfying minimum execution
                    # then we should try to utilize all those PEs
                    if (count_minimum > 1):
                        # if there are two or more PEs satisfying minimum execution
                        # populate the IDs of those PEs into a list
                        min_PE_IDs = [i for i, x in enumerate(comparison) if x == shortest_task_exec_time]
                        # then compare the number of the assigned tasks to remaining PEs
                        # and choose the one with the lowest number of assigned tasks
                        assigned_tasks = [assigned[x] for i, x in enumerate(min_PE_IDs)]
                        PE_ID_index = assigned_tasks.index(min(assigned_tasks))

                        # finally, choose the best available PE for the task
                        task.PE_ID = min_PE_IDs[PE_ID_index]
                    #   print(count_minimum, task.PE_ID)
                    else:
                        task.PE_ID = comparison.index(shortest_task_exec_time)
                    # end of if count_minimum >1:

                    # since one task is just assigned to a PE, increase the number by 1
                    assigned[task.PE_ID] += 1

                    resource_id = task.PE_ID
                    shortest_task_pe_id = resource_id
                    shortest_task = task
                    shortest_comparison = copy.deepcopy(comparison)

            # assign PE ID of the shortest task
            index = [i for i, x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            list_of_ready[index], list_of_ready[task_counter] = list_of_ready[task_counter], list_of_ready[index]
            shortest_task.PE_ID = shortest_task_pe_id

            if shortest_task.PE_ID == -1:
                print('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                      % (self.env.now, shortest_task.ID))
                print('[E] or job_**.txt file')
                assert (task.PE_ID >= 0)
            else:
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively'
                          % (self.env.now, shortest_task.ID))
                    print('%12s' % (''), comparison)
                    print('[D] Time %s: The scheduler assigns task %s to PE-%s: %s'
                          % (self.env.now, shortest_task.ID, shortest_task.PE_ID,
                             self.resource_matrix.list[shortest_task.PE_ID].name))

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            index_min_available_time = self.PEs[shortest_task.PE_ID].available_time_list.index(
                min(self.PEs[shortest_task.PE_ID].available_time_list))
            self.PEs[shortest_task.PE_ID].available_time_list[index_min_available_time] = self.env.now + \
                                                                                          shortest_comparison[
                                                                                              shortest_task.PE_ID]

            self.PEs[shortest_task.PE_ID].available_time = min(self.PEs[shortest_task.PE_ID].available_time_list)

            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list):
                if task.ID == shortest_task.ID:
                    ready_list.remove(task)

            task_counter += 1
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of while len(ready_list) > 0 :

    # end of ETF( list_of_ready)

    def OBO(self, list_of_ready):
        for task in list_of_ready:
            # if task.jobID == common.jobID_now:
                task.PE_ID = common.table[task.ID][0]
                # Optimization removed: dynamic dependencies
                task.order = common.table[task.ID][1]
                task.dynamic_dependencies = common.table[task.ID][2]
        list_of_ready.sort(key=lambda task: task.order)

    # end of HEFT(self, list_of_ready)

    def ULS(self, list_of_ready):
        # print(common.table)
        # 更新可执行队列
        for task in list_of_ready:
            for pred in task.preds:
                # 当且仅当当前任务已经分配过处理器、父结点为finish已经开始传输数据、当前任务更换处理器，才会重传。
                if task.PE_ID != -1 and task.PE_ID != common.table[task.ID][0] and getTask_10(pred) != -1:
                    task.isChange = True
                    break
            task.PE_ID = common.table[task.ID][0]
            task.order = common.table[task.ID][1]
            task.dynamic_dependencies = common.table[task.ID][2]
            task.st = common.table[task.ID][4]

        for task in common.TaskQueues.executable.list:
            for pred in task.preds:
                if task.PE_ID != -1 and task.PE_ID != common.table[task.ID][0] and getTask_10(pred) != -1:
                    task.isChange = True
                    break
            task.PE_ID = common.table[task.ID][0]
            task.order = common.table[task.ID][1]
            task.dynamic_dependencies = common.table[task.ID][2]
            task.st = common.table[task.ID][4]

        for task in common.TaskQueues.outstanding.list:
            for pred in task.preds:
                if task.PE_ID != -1 and task.PE_ID != common.table[task.ID][0] and getTask_10(pred) != -1:
                    task.isChange = True
                    break
            task.PE_ID = common.table[task.ID][0]
            task.order = common.table[task.ID][1]
            task.dynamic_dependencies = common.table[task.ID][2]
            task.st = common.table[task.ID][4]

        list_of_ready.sort(key=lambda task: task.order)
        common.TaskQueues.outstanding.list.sort(key=lambda task: task.order)
        common.TaskQueues.executable.list.sort(key=lambda task: task.order)

    def HEFT_RT(self, list_of_ready):
        if not list_of_ready:
            return
        s = common.s
        if s == 0:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 213
            elif common.job_name_temp == 'LIGO':
                deadline = 150
            elif common.job_name_temp == 'Montage':
                deadline = 120
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 873
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 237
            elif common.job_name_temp == 'Top':
                deadline = 93
            elif common.job_name_temp == 'lag_detection':
                deadline = 531
        else:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 142
            elif common.job_name_temp == 'LIGO':
                deadline = 120
            elif common.job_name_temp == 'Montage':
                deadline = 100
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 582
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 158
            elif common.job_name_temp == 'Top':
                deadline = 62
            elif common.job_name_temp == 'lag_detection':
                deadline = 354
        if common.job_name_temp == 'TEST':
            deadline = 35
        elif common.job_name_temp == 'TEST1':
            deadline = 48
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = common.arrive_time[task.jobID] + deadline
        computation_dict = {}
        power_dict = {}
        # task为就绪队列的任务
        for task in list_of_ready:
            # 对应task的computation_cost和power_dict数组
            computation_dict[task.ID] = []
            power_dict[task.ID] = []
            # cluster为cluster_list的一个集群
            for cluster in common.ClusterManager.cluster_list:
                # 当前cluster的power_consumption
                current_power = cluster.current_power_cluster
                # cluster中的PEs，是PE_ID
                for resource_idx in cluster.PE_list:
                    # 对应资源中的PE
                    resource = self.resource_matrix.list[resource_idx]
                    # task可以被该PE执行
                    if task.name in resource.supported_functionalities:
                        # 因resource.supported_functionalities.index与resource.performance.index对应
                        perf_index = resource.supported_functionalities.index(task.name)
                        computation_dict[task.ID].append(resource.performance[perf_index])
                        power_dict[task.ID].append(current_power / len(cluster.PE_list))
                    # task不可被该PE执行，赋inf
                    else:
                        computation_dict[task.ID].append(np.inf)
                        power_dict[task.ID].append(np.inf)

        # 不进入该分支
        if common.use_adaptive_scheduling:
            if common.results.job_counter == common.max_jobs_in_parallel:
                # System is oversubscribed, use EFT scheduling
                rank_metric = heftrt.RankMetric.MEAN
                op_mode = heftrt.OpMode.EFT
            else:
                # System isn't oversubscribed, use EDP scheduling
                rank_metric = heftrt.RankMetric.EDP
                op_mode = heftrt.OpMode.EDP_REL
        else:
            # MEAN
            rank_metric = heftrt.RankMetric(
                common.config.get('SCHEDULER PARAMETERS', 'heft_rankMetric', fallback='MEAN'))
            # EFT
            op_mode = heftrt.OpMode(common.config.get('SCHEDULER PARAMETERS', 'heft_opMode', fallback='EFT'))

        running_tasks = {}
        for idx in range(len(self.resource_matrix.list)):
            running_tasks[idx] = []
        # common.TaskQueues.running.list在processing_element中产生
        for task in common.TaskQueues.running.list:
            # PE
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heftrt.ScheduleEvent(task_id, task_start, task_end, proc))
        # common.TaskQueues.executable.list在DASH_Sim_core中生成
        for task in common.TaskQueues.executable.list:
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            # 当前PE的执行task不为空，task的start为PE执行列表的最后一个task的end
            if len(running_tasks[task.PE_ID]) != 0:
                task_start = running_tasks[task.PE_ID][-1].end
            # 当前PE的执行task为空，该task可放入running数组中
            else:
                task_start = self.env.now
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heftrt.ScheduleEvent(task_id, task_start, task_end, proc))
        # print(power_dict)
        start_time = int(round(time.time() * 1000))
        dict_output = heftrt.schedule_queue(
            list_of_ready,
            computation_dict=computation_dict,
            communication_matrix=common.ResourceManager.comm_band,
            time_offset=self.env.now,
            proc_schedules=running_tasks,
            rank_metric=rank_metric,
            power_dict=power_dict,
            op_mode="EFT"
        )
        end_time = int(round(time.time() * 1000))
        common.exe_time += end_time - start_time
        # print(list_of_ready)
        for task in list_of_ready:
            task.PE_ID = dict_output[task.ID][0]
            task.dynamic_dependencies = dict_output[task.ID][2]

    def HEFT_EDP(self, list_of_ready):
        if not list_of_ready:
            return
        if common.deadline_type == 'mix':
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 144
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 903
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 162
            elif common.job_name_temp == 'Top':
                deadline = 93
            elif common.job_name_temp == 'lag_detection':
                deadline = 531
        elif common.deadline_type == 'tight':
            # np.random.seed(common.iteration)
            s = common.s
            if s == 0:
                if common.job_name_temp == 'WiFi_Transmitter':
                    deadline = 216
                elif common.job_name_temp == 'WiFi_Receiver':
                    deadline = 903
                elif common.job_name_temp == 'Temporal_Mitigation':
                    deadline = 243
                elif common.job_name_temp == 'Top':
                    deadline = 93
                elif common.job_name_temp == 'lag_detection':
                    deadline = 531
            else:
                if common.job_name_temp == 'WiFi_Transmitter':
                    deadline = 144
                elif common.job_name_temp == 'WiFi_Receiver':
                    deadline = 602
                elif common.job_name_temp == 'Temporal_Mitigation':
                    deadline = 162
                elif common.job_name_temp == 'Top':
                    deadline = 62
                elif common.job_name_temp == 'lag_detection':
                    deadline = 354
        else:
            # np.random.seed(common.iteration)
            s = common.s
            if s == 0:
                if common.job_name_temp == 'WiFi_Transmitter':
                    deadline = 216
                elif common.job_name_temp == 'WiFi_Receiver':
                    deadline = 903
                elif common.job_name_temp == 'Temporal_Mitigation':
                    deadline = 243
                elif common.job_name_temp == 'Top':
                    deadline = 93
                elif common.job_name_temp == 'lag_detection':
                    deadline = 531
            else:
                if common.job_name_temp == 'WiFi_Transmitter':
                    deadline = 144
                elif common.job_name_temp == 'WiFi_Receiver':
                    deadline = 602
                elif common.job_name_temp == 'Temporal_Mitigation':
                    deadline = 162
                elif common.job_name_temp == 'Top':
                    deadline = 62
                elif common.job_name_temp == 'lag_detection':
                    deadline = 354
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = common.arrive_time[task.jobID] + deadline
        computation_dict = {}
        power_dict = {}
        # task为就绪队列的任务
        for task in list_of_ready:
            # 对应task的computation_cost和power_dict数组
            computation_dict[task.ID] = []
            power_dict[task.ID] = []
            # cluster为cluster_list的一个集群
            for cluster in common.ClusterManager.cluster_list:
                # 当前cluster的power_consumption
                current_power = cluster.current_power_cluster
                # cluster中的PEs，是PE_ID
                for resource_idx in cluster.PE_list:
                    # 对应资源中的PE
                    resource = self.resource_matrix.list[resource_idx]
                    # task可以被该PE执行
                    if task.name in resource.supported_functionalities:
                        # 因resource.supported_functionalities.index与resource.performance.index对应
                        perf_index = resource.supported_functionalities.index(task.name)
                        computation_dict[task.ID].append(resource.performance[perf_index])
                        power_dict[task.ID].append(current_power / len(cluster.PE_list))
                    # task不可被该PE执行，赋inf
                    else:
                        computation_dict[task.ID].append(np.inf)
                        power_dict[task.ID].append(np.inf)
        # 不进入该分支
        if common.use_adaptive_scheduling:
            if common.results.job_counter == common.max_jobs_in_parallel:
                # System is oversubscribed, use EFT scheduling
                rank_metric = heftrt.RankMetric.MEAN
                op_mode = heftrt.OpMode.EFT
            else:
                # System isn't oversubscribed, use EDP scheduling
                rank_metric = heftrt.RankMetric.EDP
                op_mode = heftrt.OpMode.EDP_REL
        else:
            # EDP
            rank_metric = heftrt.RankMetric('EDP')
            # EDP ABSOLUTE
            op_mode = heftrt.OpMode('EDP ABSOLUTE')

        running_tasks = {}
        for idx in range(len(self.resource_matrix.list)):
            running_tasks[idx] = []
        # common.TaskQueues.running.list在processing_element中产生
        for task in common.TaskQueues.running.list:
            # PE
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heftrt.ScheduleEvent(task_id, task_start, task_end, proc))

        # common.TaskQueues.executable.list在DASH_Sim_core中生成
        for task in common.TaskQueues.executable.list:
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            # 当前PE的执行task不为空，task的start为PE执行列表的最后一个task的end
            if len(running_tasks[task.PE_ID]) != 0:
                task_start = running_tasks[task.PE_ID][-1].end
            # 当前PE的执行task为空，该task可放入running数组中
            else:
                task_start = self.env.now
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heftrt.ScheduleEvent(task_id, task_start, task_end, proc))
        # print(power_dict)
        dict_output = heftrt.schedule_queue(
            list_of_ready,
            computation_dict=computation_dict,
            communication_matrix=common.ResourceManager.comm_band,
            time_offset=self.env.now,
            proc_schedules=running_tasks,
            rank_metric=rank_metric,
            power_dict=power_dict,
            op_mode=op_mode
        )
        # if 0 in dict_output:
        #     dict_output[0] = (4, 0, [])
        # if 1 in dict_output:
        #     dict_output[1] = (2, 0, [])
        # if 2 in dict_output:
        #     dict_output[2] = (1, 0, [])
        # if 3 in dict_output:
        #     dict_output[3] = (10, 0, [])
        # if 4 in dict_output:
        #     dict_output[4] = (5, 0, [])
        # if 5 in dict_output:
        #     dict_output[5] = (1, 0, [])
        # 1 0.07185 0.44505 -
        # 2 0.08046 0.71742 0.17979
        # 3 0.07185 0.71763 0.17946
        # 4 0.08477 0.31754 -
        # 5 0.07195 0.44552 0.18008
        # 6 0.06343 0.31738 -
        # print(dict_output)
        # print(power_dict)
        for task in list_of_ready:
            task.PE_ID = dict_output[task.ID][0]
            task.dynamic_dependencies = dict_output[task.ID][2]

    def HEFT_EDP_LB(self, list_of_ready):
        if not list_of_ready:
            return
        s = common.s
        if s == 0:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 213
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 873
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 237
            elif common.job_name_temp == 'Top':
                deadline = 93
            elif common.job_name_temp == 'lag_detection':
                deadline = 531
        else:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 142
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 582
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 158
            elif common.job_name_temp == 'Top':
                deadline = 62
            elif common.job_name_temp == 'lag_detection':
                deadline = 354
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = common.arrive_time[task.jobID] + deadline
        computation_dict = {}
        power_dict = {}
        # task为就绪队列的任务
        # for task in common.TaskQueues.outstanding.list:
        #     power_dict[task.ID] = []
        #     for cluster in common.ClusterManager.cluster_list:
        #         current_power = cluster.current_power_cluster
        #         for resource_idx in cluster.PE_list:
        #             resource = self.resource_matrix.list[resource_idx]
        #             if task.name in resource.supported_functionalities:
        #                 power_dict[task.ID].append(current_power / len(cluster.PE_list))
        #             else:
        #                 power_dict[task.ID].append(np.inf)
        for task in list_of_ready:
            # 对应task的computation_cost和power_dict数组
            computation_dict[task.ID] = []
            power_dict[task.ID] = []
            # cluster为cluster_list的一个集群
            # print(len(common.ClusterManager.cluster_list))

            for cluster in common.ClusterManager.cluster_list:
                # 当前cluster的power_consumption
                # print(cluster.name)
                # print(cluster.PE_list)
                current_power = cluster.current_power_cluster
                # cluster中的PEs，是PE_ID
                for resource_idx in cluster.PE_list:
                    # 对应资源中的PE
                    resource = self.resource_matrix.list[resource_idx]
                    # task可以被该PE执行
                    if task.name in resource.supported_functionalities:
                        # 因resource.supported_functionalities.index与resource.performance.index对应
                        perf_index = resource.supported_functionalities.index(task.name)
                        computation_dict[task.ID].append(resource.performance[perf_index])
                        power_dict[task.ID].append(current_power / len(cluster.PE_list))
                    # task不可被该PE执行，赋inf
                    else:
                        computation_dict[task.ID].append(np.inf)
                        power_dict[task.ID].append(np.inf)
        # print(power_dict)
        # print(power_dict)
        # 不进入该分支
        if common.use_adaptive_scheduling:
            if common.results.job_counter == common.max_jobs_in_parallel:
                # System is oversubscribed, use EFT scheduling
                rank_metric = heftrt.RankMetric.MEAN
                op_mode = heftrt.OpMode.EFT
            else:
                # System isn't oversubscribed, use EDP scheduling
                rank_metric = heftrt.RankMetric.EDP
                op_mode = heftrt.OpMode.EDP_REL
        else:
            # EDP
            rank_metric = heftrt.RankMetric('EDP')
            # EDP RELATIVE
            op_mode = heftrt.OpMode('EDP RELATIVE')

        running_tasks = {}
        for idx in range(len(self.resource_matrix.list)):
            running_tasks[idx] = []
        # common.TaskQueues.running.list在processing_element中产生
        for task in common.TaskQueues.running.list:
            # PE
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heftrt.ScheduleEvent(task_id, task_start, task_end, proc))

        # common.TaskQueues.executable.list在DASH_Sim_core中生成
        for task in common.TaskQueues.executable.list:
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            # 当前PE的执行task不为空，task的start为PE执行列表的最后一个task的end
            if len(running_tasks[task.PE_ID]) != 0:
                task_start = running_tasks[task.PE_ID][-1].end
            # 当前PE的执行task为空，该task可放入running数组中
            else:
                task_start = self.env.now
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heftrt.ScheduleEvent(task_id, task_start, task_end, proc))
        start_time = float(round(time.time() * 1000))
        dict_output = heftrt.schedule_queue(
            list_of_ready,
            computation_dict=computation_dict,
            communication_matrix=common.ResourceManager.comm_band,
            time_offset=self.env.now,
            proc_schedules=running_tasks,
            rank_metric=rank_metric,
            power_dict=power_dict,
            op_mode=op_mode
        )
        end_time = float(round(time.time() * 1000))
        common.exe_time += (end_time - start_time)
        for task in list_of_ready:
            task.PE_ID = dict_output[task.ID][0]
            task.dynamic_dependencies = dict_output[task.ID][2]

    def PEFT(self, list_of_ready):
        for task in list_of_ready:
            task.PE_ID = common.table[task.ID][0]
            task.order = common.table[task.ID][1]
            task.dynamic_dependencies = common.table[task.ID][2]
        list_of_ready.sort(key=lambda task: task.order)

    def PEFT_RT(self, list_of_ready):
        if not list_of_ready:
            return
        computation_dict = {}
        dag = nx.DiGraph()
        for task in list_of_ready:
            dag.add_node(task.ID)
            computation_dict[task.ID] = []
            for idx, resource in enumerate(self.resource_matrix.list):
                if task.name in resource.supported_functionalities:
                    perf_index = resource.supported_functionalities.index(task.name)
                    computation_dict[task.ID].append(resource.performance[perf_index])
                else:
                    computation_dict[task.ID].append(np.inf)
        # 对初始图进行操作，添加新的起始和终止节点
        dag = dag_merge.merge_dags(dag, merge_method=dag_merge.MergeMethod.COMMON_ENTRY_EXIT, skip_relabeling=True)
        # 给新增的起始节点和终止节点的computation_dict添加对应数量的0
        computation_dict[max(dag) - 1] = np.zeros((1, len(self.resource_matrix.list)))
        computation_dict[max(dag)] = np.zeros((1, len(self.resource_matrix.list)))
        # computation_matrix初始化，max(dag)+1为新增起始节点和终止节点后的dag节点数
        computation_matrix = np.empty((max(dag) + 1, len(self.resource_matrix.list)))

        running_tasks = {}
        for idx in range(len(self.resource_matrix.list)):
            running_tasks[idx] = []

        # common.TaskQueues.running.list在processing_element中产生
        for task in common.TaskQueues.running.list:
            # PE
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            task_start = task.start_time
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        # common.TaskQueues.executable.list在DASH_Sim_core中生成
        for task in common.TaskQueues.executable.list:
            executing_resource = self.resource_matrix.list[task.PE_ID]
            task_id = task.ID
            # 当前PE的执行task不为空，task的start为PE执行列表的最后一个task的end
            if len(running_tasks[task.PE_ID]) != 0:
                task_start = running_tasks[task.PE_ID][-1].end
            # 当前PE的执行task为空，该task可放入running数组中
            else:
                task_start = self.env.now
            task_end = task_start + executing_resource.performance[
                executing_resource.supported_functionalities.index(task.name)]
            proc = task.PE_ID
            running_tasks[proc].append(heft.ScheduleEvent(task_id, task_start, task_end, proc))

        # 将computation_dict转化为computation_matrix
        for key, val in computation_dict.items():
            computation_matrix[key, :] = val
        common.times += 1
        proc_schedules, _, dict_output = peft.schedule_dag(
            dag,
            self.PEs,
            computation_matrix=computation_matrix,
            communication_matrix=common.ResourceManager.comm_band,
            time_offset=self.env.now,
            proc_schedules=running_tasks,
            ready_queue=list_of_ready,
            relabel_nodes=False
        )
        end_time = int(round(time.time() * 1000))
        # gantt.showGanttChart(proc_schedules)
        for task in list_of_ready:
            task.PE_ID = dict_output[task.ID][0]
            task.dynamic_dependencies = dict_output[task.ID][2]

    def ILS_ETF(self, list_of_ready):
        '''
        This scheduler compares the execution times of the current
        task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the
        earliest finish time for the task. Additionally, the task with the
        lowest earliest finish time  is scheduled first
        '''
        ready_list = copy.deepcopy(list_of_ready)

        task_counter = 0

        ##################################################################################################
        ## Code added for IL by Anish
        ##################################################################################################

        ## Get minimum and maximum job ID of ready tasks
        job_id_list = []
        for task in ready_list:
            if task.jobID not in job_id_list:
                job_id_list.append(task.jobID)
            ## if task.jobID not in job_id_list :
        ## for task in list_of_ready :
        job_id_list = np.array(job_id_list)
        min_job_id = np.min(job_id_list)
        max_job_id = np.max(job_id_list)

        if common.ils_enable_dagger:
            ## Open file handles
            common.ils_open_file_handles()

            ## Print file headers
            common.ils_print_file_headers()
        ## if common.ils_enable_dagger :

        ##################################################################################################
        ## End of Code added for IL by Anish
        ##################################################################################################

        # Iterate through the list of ready tasks until all of them are scheduled
        ## while len(ready_list) > 0 :
        for shortest_task in list_of_ready:

            ## shortest_task_exec_time = np.inf
            ## shortest_task_pe_id     = -1
            ## shortest_comparison     = [np.inf] * len(self.PEs)

            ## for task in ready_list:
            ##     
            ##     comparison = [np.inf]*len(self.PEs)                                     # Initialize the comparison vector 
            ##     comm_ready = [0]*len(self.PEs)                                          # A list to store the max communication times for each PE
            ##     
            ##     if (common.DEBUG_SCH):
            ##         print ('[D] Time %s: The scheduler function is called with task %s'
            ##                %(self.env.now, task.ID))
            ##         
            ##     for i in range(len(self.resource_matrix.list)):
            ##         # if the task is supported by the resource, retrieve the index of the task
            ##         if (task.name in self.resource_matrix.list[i].supported_functionalities):
            ##             ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
            ##             
            ##                 
            ##             # $PE_comm_wait_times is a list to store the estimated communication time 
            ##             # (or the remaining communication time) of all predecessors of a task for a PE
            ##             # As simulation forwards, relevant data is being sent after a task is completed
            ##             # based on the time instance, one should consider either whole communication
            ##             # time or the remaining communication time for scheduling
            ##             PE_comm_wait_times = []
            ##             
            ##             # $PE_wait_time is a list to store the estimated wait times for a PE
            ##             # till that PE is available if the PE is currently running a task
            ##             PE_wait_time = []
            ##               
            ##             job_ID = -1                                                     # Initialize the job ID
            ##             
            ##             # Retrieve the job ID which the current task belongs to
            ##             for ii, job in enumerate(self.jobs.list):
            ##                 if job.name == task.jobname:
            ##                     job_ID = ii
            ##                     
            ##             for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
            ##                 # data required from the predecessor for $ready_task
            ##                 c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]
            ##                 
            ##                 # retrieve the real ID  of the predecessor based on the job ID
            ##                 real_predecessor_ID = predecessor + task.ID - task.base_ID
            ##                 
            ##                 # Initialize following two variables which will be used if 
            ##                 # PE to PE communication is utilized
            ##                 predecessor_PE_ID = -1
            ##                 predecessor_finish_time = -1
            ##                 
            ##                 
            ##                 for completed in common.TaskQueues.completed.list:
            ##                     if (completed.ID == real_predecessor_ID):
            ##                         predecessor_PE_ID = completed.PE_ID
            ##                         predecessor_finish_time = completed.finish_time
            ##                         #print(predecessor, predecessor_finish_time, predecessor_PE_ID)
            ##                         
            ##                 
            ##                 if (common.PE_to_PE):
            ##                     # Compute the PE to PE communication time
            ##                     #PE_to_PE_band = self.resource_matrix.comm_band[predecessor_PE_ID, i]
            ##                     PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
            ##                     PE_to_PE_comm_time = int(c_vol/PE_to_PE_band)
            ##                     
            ##                     PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
            ##                     
            ##                     if (common.DEBUG_SCH):
            ##                         print('[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d' 
            ##                               %(self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
            ##                     
            ##                 if (common.shared_memory):
            ##                     # Compute the communication time considering the shared memory
            ##                     # only consider memory to PE communication time
            ##                     # since the task passed the 1st phase (PE to memory communication)
            ##                     # and its status changed to ready 
            ##                     
            ##                     #PE_to_memory_band = self.resource_matrix.comm_band[predecessor_PE_ID, -1]
            ##                     memory_to_PE_band = common.ResourceManager.comm_band[self.resource_matrix.list[-1].ID, i]
            ##                     shared_memory_comm_time = int(c_vol/memory_to_PE_band)
            ##                     
            ##                     PE_comm_wait_times.append(shared_memory_comm_time)
            ##                     if (common.DEBUG_SCH):
            ##                         print('[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d' 
            ##                               %(self.env.now, i, real_predecessor_ID, task.ID, PE_comm_wait_times[-1]))
            ##                 
            ##                 # $comm_ready contains the estimated communication time 
            ##                 # for the resource in consideration for scheduling
            ##                 # maximum value is chosen since it represents the time required for all
            ##                 # data becomes available for the resource. 
            ##                 comm_ready[i] = (max(PE_comm_wait_times))
            ##                 
            ##             # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors: 
            ##             
            ##             # if a resource currently is executing a task, then the estimated remaining time
            ##             # for the task completion should be considered during scheduling
            ##             PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))
            ##             
            ##             # update the comparison vector accordingly    
            ##             comparison[i] = self.resource_matrix.list[i].performance[ind] + max(comm_ready[i], PE_wait_time[-1])
            ##             

            ##             # after going over each resource, choose the one which gives the minimum result
            ##             resource_id = comparison.index(min(comparison))
            ##             #print('aa',comparison)
            ##         # end of if (task.name in self.resource_matrix.list[i]...
            ##         
            ##     # obtain the task ID, resource for the task with earliest finish time 
            ##     # based on the computation 
            ##     #print('bb',comparison)
            ##     if min(comparison) < shortest_task_exec_time :
            ##         shortest_task_exec_time = min(comparison)
            ##         shortest_task_pe_id     = resource_id
            ##         shortest_task           = task
            ##         shortest_comparison     = comparison

            ##         
            ##     # end of for i in range(len(self.resource_matrix.list)):
            ## # end of for task in ready_list:
            ## 
            ## # assign PE ID of the shortest task 
            ## index = [i for i,x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            ## list_of_ready[index].PE_ID = shortest_task_pe_id
            ## list_of_ready[index], list_of_ready[task_counter] = list_of_ready[task_counter], list_of_ready[index]
            ## shortest_task.PE_ID        = shortest_task_pe_id

            ## if shortest_task.PE_ID == -1:
            ##     print ('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
            ##            % (self.env.now, shortest_task.ID))
            ##     print ('[E] or job_**.txt file')
            ##     assert(task.PE_ID >= 0)           
            ## else: 
            ##     if (common.DEBUG_SCH):
            ##         print('[D] Time %s: Estimated execution times for each PE with task %s, respectively' 
            ##                   %(self.env.now, shortest_task.ID))
            ##         print('%12s'%(''), comparison)
            ##         print ('[D] Time %s: The scheduler assigns task %s to PE-%s: %s'
            ##                %(self.env.now, shortest_task.ID, shortest_task.PE_ID, 
            ##                  self.resource_matrix.list[shortest_task.PE_ID].name))

            ##################################################################################################
            ## Code added for IL by Anish
            ##################################################################################################

            ## Collect cluster available times
            cluster_free_times = []

            ## Append the cluster available times to the list of features
            cluster0_free_times = [max(self.PEs[0].available_time - self.env.now, 0), \
                                   max(self.PEs[1].available_time - self.env.now, 0), \
                                   max(self.PEs[2].available_time - self.env.now, 0), \
                                   max(self.PEs[3].available_time - self.env.now, 0), \
                                   ]
            cluster1_free_times = [max(self.PEs[4].available_time - self.env.now, 0), \
                                   max(self.PEs[5].available_time - self.env.now, 0), \
                                   max(self.PEs[6].available_time - self.env.now, 0), \
                                   max(self.PEs[7].available_time - self.env.now, 0), \
                                   ]
            cluster2_free_times = [max(self.PEs[8].available_time - self.env.now, 0), \
                                   max(self.PEs[9].available_time - self.env.now, 0), \
                                   ]
            cluster3_free_times = [max(self.PEs[10].available_time - self.env.now, 0), \
                                   max(self.PEs[11].available_time - self.env.now, 0), \
                                   max(self.PEs[12].available_time - self.env.now, 0), \
                                   max(self.PEs[13].available_time - self.env.now, 0), \
                                   ]
            cluster4_free_times = [max(self.PEs[14].available_time - self.env.now, 0), \
                                   max(self.PEs[15].available_time - self.env.now, 0), \
                                   ]

            cluster_free_times.append(np.min(cluster0_free_times))
            cluster_free_times.append(np.min(cluster1_free_times))
            cluster_free_times.append(np.min(cluster2_free_times))
            cluster_free_times.append(np.min(cluster3_free_times))
            cluster_free_times.append(np.min(cluster4_free_times))
            cluster_free_times = np.array(cluster_free_times) / 30

            cluster0_free_times = np.array(cluster0_free_times) / 30
            cluster1_free_times = np.array(cluster1_free_times) / 30
            cluster2_free_times = np.array(cluster2_free_times) / 30
            cluster3_free_times = np.array(cluster3_free_times) / 30
            cluster4_free_times = np.array(cluster4_free_times) / 30
            cluster0_free_times = np.pad(cluster0_free_times, (0, (5 - len(cluster0_free_times))), constant_values=(10),
                                         mode='constant')
            cluster1_free_times = np.pad(cluster1_free_times, (0, (5 - len(cluster1_free_times))), constant_values=(10),
                                         mode='constant')
            cluster2_free_times = np.pad(cluster2_free_times, (0, (5 - len(cluster2_free_times))), constant_values=(10),
                                         mode='constant')
            cluster3_free_times = np.pad(cluster3_free_times, (0, (5 - len(cluster3_free_times))), constant_values=(10),
                                         mode='constant')
            cluster4_free_times = np.pad(cluster4_free_times, (0, (5 - len(cluster4_free_times))), constant_values=(10),
                                         mode='constant')

            ##################################################################################################
            ## End of Code added for IL by Anish
            ##################################################################################################

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            ## self.PEs[shortest_task.PE_ID].available_time = self.env.now + shortest_comparison[shortest_task.PE_ID]

            ##################################################################################################
            ## Code added for IL by Anish
            ##################################################################################################
            il_features = []

            ## Populate execution time profile (cluster-wise)
            task_exec_times = []
            for PE in self.PEs:
                ## Skip analyzing resources if resource is MEMORY or CACHE
                if 'MEM' in PE.name or 'CAC' in PE.name:
                    continue

                if PE.ID == 0 or PE.ID == 4 or PE.ID == 8 or PE.ID == 10 or PE.ID == 14:
                    if shortest_task.name in self.resource_matrix.list[PE.ID].supported_functionalities:
                        ind = self.resource_matrix.list[PE.ID].supported_functionalities.index(shortest_task.name)
                        exec_time = self.resource_matrix.list[PE.ID].performance[ind]
                        task_exec_times.append(exec_time)
                    else:
                        task_exec_times.append(10000)
                    ## if shortest_task.name in self.resource_matrix.list[PE.ID].supported_functionalities :
                ## if PE.ID == 0 or PE.ID == 4 or PE.ID == 8 or PE.ID == 10 or PE.ID == 14:
            ## for PE in self.PEs :

            ## Normalize cluster execution times
            task_exec_times = np.array(task_exec_times)
            if len(task_exec_times[np.all([task_exec_times != 0])]) == 0:
                normalized_task_exec_times = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
            else:
                normalized_task_exec_times = common.get_normalized_list(task_exec_times)
            ## if len(task_exec_times[np.all([task_exec_times != 0, task_exec_times != 10000])]) == 0 :

            ## Populate normalized downward depth of task in DAG
            task_depth = shortest_task.dag_depth
            task_job_index = -1
            for job_index, job in enumerate(self.jobs.list):
                if shortest_task.jobname == job.name:
                    job_depth = self.jobs.list[job_index].dag_depth['DAG']
                    num_tasks = len(self.jobs.list[job_index].task_list)
                    task_job_index = job_index
            normalized_task_depth = task_depth / job_depth

            ## Populate relative job ID
            if max_job_id - min_job_id == 0:
                relative_job_id = 0
            else:
                relative_job_id = (shortest_task.jobID - min_job_id) / (max_job_id - min_job_id)
            ## if max_job_id - min_job_id == 0 :

            pred_task_comm_time_list = []
            predecessor_PEs = []

            ## Populate predecessor PE IDs
            task_preds = np.array(shortest_task.preds)
            task_pred_IDs = task_preds + shortest_task.head_ID
            task_preds_cluster_list = []
            task_preds_comm_vol = []
            for task_pred in task_pred_IDs:
                for completed_task in common.TaskQueues.completed.list:
                    if completed_task.ID == task_pred:
                        task_preds_cluster_list.append(common.get_cluster(completed_task.PE_ID))
                        c_vol = self.jobs.list[task_job_index].comm_vol[
                            task_pred - shortest_task.head_ID, shortest_task.base_ID]
                        task_preds_comm_vol.append(c_vol / 1000)
                        ## for PE in self.PEs :
                    ## if completed_task.ID == task_pred :
                ## for completed_task in common.TaskQueues.completed.list :
            ## for task_pred in task_preds :

            num_short_preds = 5 - len(shortest_task.preds)
            for num in range(num_short_preds):
                task_preds = np.append(task_preds, 10000)
                task_preds_cluster_list.append(10000)
                task_preds_comm_vol.append(10000)
            ## for num in range(num_short_preds) :

            task_preds_cluster_list = np.array(task_preds_cluster_list)
            task_preds_cluster_list = task_preds_cluster_list / 4
            task_preds_cluster_list[task_preds_cluster_list > 1000] = 10
            task_preds[task_preds > 1000] = 50
            task_preds_comm_vol = np.array(task_preds_comm_vol)
            task_preds_comm_vol[task_preds_comm_vol > 1000] = 10

            # Populate job type
            if shortest_task.jobname == 'WiFi_Transmitter':
                job_type = 0
            if shortest_task.jobname == 'WiFi_Receiver':
                job_type = 1
            if shortest_task.jobname == 'lag_detection':
                job_type = 2
            if shortest_task.jobname == 'SC-T':
                job_type = 3
            if shortest_task.jobname == 'SC-R':
                job_type = 4
            if shortest_task.jobname == 'Temporal_Mitigation':
                job_type = 5

            # Populate normalized task ID
            normalized_task_ID = shortest_task.base_ID / num_tasks

            # Populate features and save samples
            il_features.append(normalized_task_ID)
            il_features.extend(normalized_task_exec_times)
            il_features.append(normalized_task_depth)
            il_features.append(relative_job_id)
            il_features.append(job_type)
            il_features.extend(task_preds)
            il_features.extend(task_preds_cluster_list)
            il_features.extend(task_preds_comm_vol)

            ## Process features to feed to model
            features_to_cluster_model = cluster_free_times
            features_to_cluster_model = np.append(features_to_cluster_model, np.array(il_features)).reshape(1, -1)
            # features_to_cluster_model = features_to_cluster_model.reshape(1, -1)

            ## Pass features to the model and predict cluster
            task_cluster_prediction = int(np.around(common.il_clustera_model.predict(features_to_cluster_model)[0]))

            # Fail-safe mechanism
            if task_cluster_prediction == 0:
                test_PE_index = 0
            if task_cluster_prediction == 1:
                test_PE_index = 4
            if task_cluster_prediction == 2:
                test_PE_index = 8
            if task_cluster_prediction == 3:
                test_PE_index = 10
            if task_cluster_prediction == 4:
                test_PE_index = 14

            if not shortest_task.name in self.resource_matrix.list[test_PE_index].supported_functionalities:
                if job_type == 5:
                    task_cluster_prediction = 0
                else:
                    task_cluster_prediction = 1
            ## if not shortest_task.name in  self.resource_matrix.list[test_PE_index].supported_functionalities :

            ## Pass features to the model and predict PE within cluster
            if task_cluster_prediction == 0:
                features_to_task_model = cluster0_free_times
                features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                task_PE_prediction = int(np.around(common.il_cluster0_model.predict(features_to_task_model)[0]))
                if task_PE_prediction < 0 or task_PE_prediction > 3:
                    cluster_free_times = cluster0_free_times[0:4]
                    task_PE_prediction = np.argmin(cluster_free_times)
                task_PE_prediction += 0
            if task_cluster_prediction == 1:
                features_to_task_model = cluster1_free_times
                features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                task_PE_prediction = int(np.around(common.il_cluster1_model.predict(features_to_task_model)[0]))
                if task_PE_prediction < 0 or task_PE_prediction > 3:
                    cluster_free_times = cluster1_free_times[0:4]
                    task_PE_prediction = np.argmin(cluster_free_times)
                task_PE_prediction += 4
            if task_cluster_prediction == 2:
                features_to_task_model = cluster2_free_times
                features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                task_PE_prediction = int(np.around(common.il_cluster2_model.predict(features_to_task_model)[0]))
                if task_PE_prediction < 0 or task_PE_prediction > 1:
                    cluster_free_times = cluster2_free_times[0:2]
                    task_PE_prediction = np.argmin(cluster_free_times)
                task_PE_prediction += 8
            if task_cluster_prediction == 3:
                features_to_task_model = cluster3_free_times
                features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                task_PE_prediction = int(np.around(common.il_cluster3_model.predict(features_to_task_model)[0]))
                if task_PE_prediction < 0 or task_PE_prediction > 3:
                    cluster_free_times = cluster3_free_times[0:4]
                    task_PE_prediction = np.argmin(cluster_free_times)
                task_PE_prediction += 10
            if task_cluster_prediction == 4:
                features_to_task_model = cluster4_free_times
                features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                task_PE_prediction = int(np.around(common.il_cluster4_model.predict(features_to_task_model)[0]))
                if task_PE_prediction < 0 or task_PE_prediction > 1:
                    cluster_free_times = cluster4_free_times[0:2]
                    task_PE_prediction = np.argmin(cluster_free_times)
                task_PE_prediction += 14

            ## Assign predicted PE to task
            shortest_task.PE_ID = task_PE_prediction

            ## Update PE available time
            PE_comm_wait_times = []

            for predecessor in shortest_task.preds:
                # data required from the predecessor for $ready_task
                for ii, job in enumerate(self.jobs.list):
                    if job.name == shortest_task.jobname:
                        job_ID = ii
                c_vol = self.jobs.list[job_ID].comm_vol[predecessor, shortest_task.base_ID]

                # retrieve the real ID  of the predecessor based on the job ID
                real_predecessor_ID = predecessor + shortest_task.ID - shortest_task.base_ID

                # Initialize following two variables which will be used if 
                # PE to PE communication is utilized
                predecessor_PE_ID = -1
                predecessor_finish_time = -1

                for completed in common.TaskQueues.completed.list:
                    if completed.ID == real_predecessor_ID:
                        predecessor_PE_ID = completed.PE_ID
                        predecessor_finish_time = completed.finish_time
                        # print(predecessor, predecessor_finish_time, predecessor_PE_ID)

                if (common.PE_to_PE):
                    # Compute the PE to PE communication time
                    PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, shortest_task.PE_ID]
                    PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                PE_comm_wait_times.append(max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
            ## for predecessor in task.preds :

            predecessor_ready_time = max(PE_comm_wait_times, default=0)
            PE_wait_time = max((self.PEs[shortest_task.PE_ID].available_time - self.env.now), 0)
            # update the comparison vector accordingly    
            ind = self.resource_matrix.list[shortest_task.PE_ID].supported_functionalities.index(shortest_task.name)
            expected_latency = self.resource_matrix.list[shortest_task.PE_ID].performance[ind] + max(
                predecessor_ready_time, PE_wait_time)

            ## Enable data aggregation if flag is set to active
            if common.ils_enable_dagger == True:

                ##################################################################################################
                ## Online-ETF to construct on-the-fly Oracle
                ##################################################################################################
                comparison = [np.inf] * len(self.PEs)  # Initialize the comparison vector
                comm_ready = [0] * len(self.PEs)  # A list to store the max communication times for each PE

                for i in range(len(self.resource_matrix.list)):
                    # if the task is supported by the resource, retrieve the index of the task
                    if (shortest_task.name in self.resource_matrix.list[i].supported_functionalities):
                        ind = self.resource_matrix.list[i].supported_functionalities.index(shortest_task.name)

                        # $PE_comm_wait_times is a list to store the estimated communication time 
                        # (or the remaining communication time) of all predecessors of a task for a PE
                        # As simulation forwards, relevant data is being sent after a task is completed
                        # based on the time instance, one should consider either whole communication
                        # time or the remaining communication time for scheduling
                        PE_comm_wait_times = []

                        # $PE_wait_time is a list to store the estimated wait times for a PE
                        # till that PE is available if the PE is currently running a task
                        PE_wait_time = []

                        job_ID = -1  # Initialize the job ID

                        # Retrieve the job ID which the current task belongs to
                        for ii, job in enumerate(self.jobs.list):
                            if job.name == shortest_task.jobname:
                                job_ID = ii

                        for predecessor in self.jobs.list[job_ID].task_list[shortest_task.base_ID].predecessors:
                            # data required from the predecessor for $ready_task
                            c_vol = self.jobs.list[job_ID].comm_vol[predecessor, shortest_task.base_ID]

                            # retrieve the real ID  of the predecessor based on the job ID
                            real_predecessor_ID = predecessor + shortest_task.ID - shortest_task.base_ID

                            # Initialize following two variables which will be used if 
                            # PE to PE communication is utilized
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1

                            for completed in common.TaskQueues.completed.list:
                                if (completed.ID == real_predecessor_ID):
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                                    # print(predecessor, predecessor_finish_time, predecessor_PE_ID)

                            if (common.PE_to_PE):
                                # Compute the PE to PE communication time
                                # PE_to_PE_band = self.resource_matrix.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                                PE_comm_wait_times.append(
                                    max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))

                                if (common.DEBUG_SCH):
                                    print(
                                        '[D] Time %s: Estimated communication time between PE-%s to PE-%s from task %s to task %s is %d'
                                        % (self.env.now, predecessor_PE_ID, i, real_predecessor_ID, shortest_task.ID,
                                           PE_comm_wait_times[-1]))

                            if (common.shared_memory):
                                # Compute the communication time considering the shared memory
                                # only consider memory to PE communication time
                                # since the task passed the 1st phase (PE to memory communication)
                                # and its status changed to ready 

                                # PE_to_memory_band = self.resource_matrix.comm_band[predecessor_PE_ID, -1]
                                memory_to_PE_band = common.ResourceManager.comm_band[
                                    self.resource_matrix.list[-1].ID, i]
                                shared_memory_comm_time = int(c_vol / memory_to_PE_band)

                                PE_comm_wait_times.append(shared_memory_comm_time)
                                if (common.DEBUG_SCH):
                                    print(
                                        '[D] Time %s: Estimated communication time between memory to PE-%s from task %s to task %s is %d'
                                        % (
                                            self.env.now, i, real_predecessor_ID, shortest_task.ID,
                                            PE_comm_wait_times[-1]))

                            # $comm_ready contains the estimated communication time 
                            # for the resource in consideration for scheduling
                            # maximum value is chosen since it represents the time required for all
                            # data becomes available for the resource. 
                            comm_ready[i] = (max(PE_comm_wait_times))

                        # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors: 

                        # if a resource currently is executing a task, then the estimated remaining time
                        # for the task completion should be considered during scheduling
                        PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                        # update the comparison vector accordingly    
                        comparison[i] = self.resource_matrix.list[i].performance[ind] + max(comm_ready[i],
                                                                                            PE_wait_time[-1])

                        # after going over each resource, choose the one which gives the minimum result
                        resource_id = comparison.index(min(comparison))
                        # print('aa',comparison)
                    # end of if (task.name in self.resource_matrix.list[i]...
                ## for i in range(len(self.resource_matrix.list)):

                resource_label = resource_id
                cluster_label = common.get_cluster(resource_label)

                ## Initialize flag to decide if PE sample should be aggregated
                aggregate_PE_sample = 0

                ## Check if cluster predict matches Oracle
                if cluster_label != task_cluster_prediction:
                    ## Write data onto file 
                    common.ils_clustera_fp.write(str(self.env.now) + ',')
                    common.ils_clustera_fp.write(str(shortest_task.ID))
                    for feature in cluster_free_times:
                        common.ils_clustera_fp.write(',' + str(feature))
                    for feature in il_features:
                        common.ils_clustera_fp.write(',' + str(feature))
                    common.ils_clustera_fp.write(',' + str(resource_label))
                    common.ils_clustera_fp.write(',' + str(cluster_label))
                    common.ils_clustera_fp.write('\n')

                    ## Check for PE prediction with correct cluster
                    ## Previously we were searching in the incorrect cluster space
                    ## Pass features to the model and predict PE within cluster
                    if cluster_label == 0:
                        features_to_task_model = cluster0_free_times
                        features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                        task_PE_prediction = int(np.around(common.il_cluster0_model.predict(features_to_task_model)[0]))
                        if task_PE_prediction < 0 or task_PE_prediction > 3:
                            cluster_free_times = cluster0_free_times[0:4]
                            task_PE_prediction = np.argmin(cluster_free_times)
                        task_PE_prediction += 0
                    if cluster_label == 1:
                        features_to_task_model = cluster1_free_times
                        features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                        task_PE_prediction = int(np.around(common.il_cluster1_model.predict(features_to_task_model)[0]))
                        if task_PE_prediction < 0 or task_PE_prediction > 3:
                            cluster_free_times = cluster1_free_times[0:4]
                            task_PE_prediction = np.argmin(cluster_free_times)
                        task_PE_prediction += 4
                    if cluster_label == 2:
                        features_to_task_model = cluster2_free_times
                        features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                        task_PE_prediction = int(np.around(common.il_cluster2_model.predict(features_to_task_model)[0]))
                        if task_PE_prediction < 0 or task_PE_prediction > 1:
                            cluster_free_times = cluster2_free_times[0:2]
                            task_PE_prediction = np.argmin(cluster_free_times)
                        task_PE_prediction += 8
                    if cluster_label == 3:
                        features_to_task_model = cluster3_free_times
                        features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                        task_PE_prediction = int(np.around(common.il_cluster3_model.predict(features_to_task_model)[0]))
                        if task_PE_prediction < 0 or task_PE_prediction > 3:
                            cluster_free_times = cluster3_free_times[0:4]
                            task_PE_prediction = np.argmin(cluster_free_times)
                        task_PE_prediction += 10
                    if cluster_label == 4:
                        features_to_task_model = cluster4_free_times
                        features_to_task_model = np.append(features_to_task_model, np.array(il_features)).reshape(1, -1)
                        task_PE_prediction = int(np.around(common.il_cluster4_model.predict(features_to_task_model)[0]))
                        if task_PE_prediction < 0 or task_PE_prediction > 1:
                            cluster_free_times = cluster4_free_times[0:2]
                            task_PE_prediction = np.argmin(cluster_free_times)
                        task_PE_prediction += 14

                    ## Set flag to aggregate PE sample if PE prediction is incorrect
                    if task_PE_prediction != resource_label:
                        aggregate_PE_sample = 1
                else:
                    if task_PE_prediction != resource_label:
                        aggregate_PE_sample = 1
                ## if cluster_label != task_cluster_prediction :

                if aggregate_PE_sample == 1:
                    if cluster_label == 0:
                        common.ils_cluster0_fp.write(str(self.env.now) + ',')
                        common.ils_cluster0_fp.write(str(shortest_task.ID))
                        for feature in cluster0_free_times:
                            common.ils_cluster0_fp.write(',' + str(feature))
                        for feature in il_features:
                            common.ils_cluster0_fp.write(',' + str(feature))
                        common.ils_cluster0_fp.write(',' + str(resource_label))
                        common.ils_cluster0_fp.write(',' + str(cluster_label))
                        common.ils_cluster0_fp.write('\n')
                    if cluster_label == 1:
                        common.ils_cluster1_fp.write(str(self.env.now) + ',')
                        common.ils_cluster1_fp.write(str(shortest_task.ID))
                        for feature in cluster1_free_times:
                            common.ils_cluster1_fp.write(',' + str(feature))
                        for feature in il_features:
                            common.ils_cluster1_fp.write(',' + str(feature))
                        common.ils_cluster1_fp.write(',' + str(resource_label))
                        common.ils_cluster1_fp.write(',' + str(cluster_label))
                        common.ils_cluster1_fp.write('\n')
                    if cluster_label == 2:
                        common.ils_cluster2_fp.write(str(self.env.now) + ',')
                        common.ils_cluster2_fp.write(str(shortest_task.ID))
                        for feature in cluster2_free_times:
                            common.ils_cluster2_fp.write(',' + str(feature))
                        for feature in il_features:
                            common.ils_cluster2_fp.write(',' + str(feature))
                        common.ils_cluster2_fp.write(',' + str(resource_label))
                        common.ils_cluster2_fp.write(',' + str(cluster_label))
                        common.ils_cluster2_fp.write('\n')
                    if cluster_label == 3:
                        common.ils_cluster3_fp.write(str(self.env.now) + ',')
                        common.ils_cluster3_fp.write(str(shortest_task.ID))
                        for feature in cluster3_free_times:
                            common.ils_cluster3_fp.write(',' + str(feature))
                        for feature in il_features:
                            common.ils_cluster3_fp.write(',' + str(feature))
                        common.ils_cluster3_fp.write(',' + str(resource_label))
                        common.ils_cluster3_fp.write(',' + str(cluster_label))
                        common.ils_cluster3_fp.write('\n')
                    if cluster_label == 4:
                        common.ils_cluster4_fp.write(str(self.env.now) + ',')
                        common.ils_cluster4_fp.write(str(shortest_task.ID))
                        for feature in cluster4_free_times:
                            common.ils_cluster4_fp.write(',' + str(feature))
                        for feature in il_features:
                            common.ils_cluster4_fp.write(',' + str(feature))
                        common.ils_cluster4_fp.write(',' + str(resource_label))
                        common.ils_cluster4_fp.write(',' + str(cluster_label))
                        common.ils_cluster4_fp.write('\n')
                ## if aggregate_PE_sample == 1 :

                ##################################################################################################
                ## End of Online-ETF to construct on-the-fly Oracle
                ##################################################################################################

            ## if common.ils_enable_dagger == True :

            ## Update PE available time after online-ETF computation
            self.PEs[shortest_task.PE_ID].available_time = self.env.now + expected_latency

            ##################################################################################################
            ## End of Code added for IL by Anish
            ##################################################################################################

            # Remove the task which got a schedule successfully
            ## for i, task in enumerate(ready_list) :
            ##     if task.ID == shortest_task.ID :
            ##         ready_list.remove(task)

            task_counter += 1
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of while len(ready_list) > 0 :

        ## Close file handles
        if common.ils_enable_dataset_save or common.ils_enable_dagger:
            common.ils_close_file_handles()
        ## if common.ils_enable_dataset_save :

    # end of ETF( list_of_ready)

    def DPDS(self, list_of_ready):
        '''
        This scheduler compares the execution times of the current
        task for available resources and also considers if a resource has
        already a task running. it picks the resource which will give the
        earliest finish time for the task. Additionally, the task with the
        lowest earliest finish time  is scheduled first
        '''
        if not list_of_ready:
            return
        s = common.s
        common.times += 1
        if s == 0:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 213
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 873
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 237
            elif common.job_name_temp == 'Top':
                deadline = 93
            elif common.job_name_temp == 'lag_detection':
                deadline = 531
        else:
            if common.job_name_temp == 'WiFi_Transmitter':
                deadline = 142
            elif common.job_name_temp == 'WiFi_Receiver':
                deadline = 582
            elif common.job_name_temp == 'Temporal_Mitigation':
                deadline = 158
            elif common.job_name_temp == 'Top':
                deadline = 62
            elif common.job_name_temp == 'lag_detection':
                deadline = 354
        if common.job_name_temp == 'TEST':
            deadline = 35
        elif common.job_name_temp == 'TEST1':
            deadline = 48
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = common.arrive_time[task.jobID] + deadline
        ready_list = copy.deepcopy(list_of_ready)

        task_counter = 0

        ##################################################################################################
        ## IL-Scheduler
        ##################################################################################################

        if common.ils_enable_dataset_save:
            ## Open file handles
            common.ils_open_file_handles()

            ## Print file headers
            common.ils_print_file_headers()
        ## if common.ils_enable_dataset_save or common.ils_enable_policy_decision :

        if common.ils_enable_dataset_save or common.ils_enable_policy_decision:
            ## Get minimum and maximum job ID of ready tasks
            job_id_list = []
            for task in ready_list:
                if task.jobID not in job_id_list:
                    job_id_list.append(task.jobID)
                ## if task.jobID not in job_id_list :
            ## for task in list_of_ready :
            job_id_list = np.array(job_id_list)
            min_job_id = np.min(job_id_list)
            max_job_id = np.max(job_id_list)
        ## if common.ils_enable_dataset_save or common.ils_enable_policy_decision :

        ##################################################################################################
        ## End of IL-Scheduler
        ##################################################################################################

        # Iterate through the list of ready tasks until all of them are scheduled
        while len(ready_list) > 0:

            shortest_task_exec_time = np.inf
            shortest_task_pe_id = -1
            shortest_comparison = [np.inf] * len(self.PEs)
            start_time = int(round(time.time() * 1000))
            for task in ready_list:
                comparison = [np.inf] * len(self.PEs)  # Initialize the comparison vector
                comm_ready = [0] * len(self.PEs)  # A list to store the max communication times for each PE
                for i in range(len(self.resource_matrix.list)):
                    if (task.name in self.resource_matrix.list[i].supported_functionalities):
                        ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)
                        PE_comm_wait_times = []
                        PE_wait_time = []

                        job_ID = -1  # Initialize the job ID
                        for ii, job in enumerate(self.jobs.list):
                            if job.name == task.jobname:
                                job_ID = ii

                        for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                            # data required from the predecessor for $ready_task
                            c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]
                            real_predecessor_ID = predecessor + task.ID - task.base_ID
                            predecessor_PE_ID = -1
                            predecessor_finish_time = -1

                            for completed in common.TaskQueues.completed.list:
                                if (completed.ID == real_predecessor_ID):
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time

                            PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                            PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                            PE_comm_wait_times.append(
                            max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))
                            comm_ready[i] = (max(PE_comm_wait_times))
                        PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))
                        comparison[i] = self.resource_matrix.list[i].performance[ind] + max(comm_ready[i],
                                                                                            PE_wait_time[-1])
                        resource_id = comparison.index(min(comparison))
                if min(comparison) < shortest_task_exec_time:
                    shortest_task_exec_time = min(comparison)
                    shortest_task_pe_id = resource_id
                    shortest_task = task
                    shortest_comparison = comparison

                # end of for i in range(len(self.resource_matrix.list)):
            # end of for task in ready_list:
            end_time = int(round(time.time() * 1000))
            common.exe_time += end_time - start_time
            # assign PE ID of the shortest task 
            index = [i for i, x in enumerate(list_of_ready) if x.ID == shortest_task.ID][0]
            list_of_ready[index].PE_ID = shortest_task_pe_id
            list_of_ready[index], list_of_ready[task_counter] = list_of_ready[task_counter], list_of_ready[index]
            shortest_task.PE_ID = shortest_task_pe_id


            if shortest_task.PE_ID == -1:
                print('[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                      % (self.env.now, shortest_task.ID))
                print('[E] or job_**.txt file')
                assert (task.PE_ID >= 0)
            else:
                if (common.DEBUG_SCH):
                    print('[D] Time %s: Estimated execution times for each PE with task %s, respectively'
                          % (self.env.now, shortest_task.ID))
                    print('%12s' % (''), comparison)
                    print('[D] Time %s: The scheduler assigns task %s to PE-%s: %s'
                          % (self.env.now, shortest_task.ID, shortest_task.PE_ID,
                             self.resource_matrix.list[shortest_task.PE_ID].name))

            ##################################################################################################
            ## IL-Scheduler
            ##################################################################################################

            if common.ils_enable_dataset_save:

                ## Collect cluster available times
                cluster_free_times = []

                ## Append the cluster available times to the list of features
                cluster0_free_times = [max(self.PEs[0].available_time - self.env.now, 0), \
                                       max(self.PEs[1].available_time - self.env.now, 0), \
                                       max(self.PEs[2].available_time - self.env.now, 0), \
                                       max(self.PEs[3].available_time - self.env.now, 0), \
                                       ]
                cluster1_free_times = [max(self.PEs[4].available_time - self.env.now, 0), \
                                       max(self.PEs[5].available_time - self.env.now, 0), \
                                       max(self.PEs[6].available_time - self.env.now, 0), \
                                       max(self.PEs[7].available_time - self.env.now, 0), \
                                       ]
                cluster2_free_times = [max(self.PEs[8].available_time - self.env.now, 0), \
                                       max(self.PEs[9].available_time - self.env.now, 0), \
                                       ]
                cluster3_free_times = [max(self.PEs[10].available_time - self.env.now, 0), \
                                       max(self.PEs[11].available_time - self.env.now, 0), \
                                       max(self.PEs[12].available_time - self.env.now, 0), \
                                       max(self.PEs[13].available_time - self.env.now, 0), \
                                       ]
                cluster4_free_times = [max(self.PEs[14].available_time - self.env.now, 0), \
                                       max(self.PEs[15].available_time - self.env.now, 0), \
                                       ]

                cluster_free_times.append(np.min(cluster0_free_times))
                cluster_free_times.append(np.min(cluster1_free_times))
                cluster_free_times.append(np.min(cluster2_free_times))
                cluster_free_times.append(np.min(cluster3_free_times))
                cluster_free_times.append(np.min(cluster4_free_times))
                cluster_free_times = np.array(cluster_free_times) / 30

                cluster0_free_times = np.array(cluster0_free_times) / 30
                cluster1_free_times = np.array(cluster1_free_times) / 30
                cluster2_free_times = np.array(cluster2_free_times) / 30
                cluster3_free_times = np.array(cluster3_free_times) / 30
                cluster4_free_times = np.array(cluster4_free_times) / 30
                cluster0_free_times = np.pad(cluster0_free_times, (0, (5 - len(cluster0_free_times))),
                                             constant_values=(10), mode='constant')
                cluster1_free_times = np.pad(cluster1_free_times, (0, (5 - len(cluster1_free_times))),
                                             constant_values=(10), mode='constant')
                cluster2_free_times = np.pad(cluster2_free_times, (0, (5 - len(cluster2_free_times))),
                                             constant_values=(10), mode='constant')
                cluster3_free_times = np.pad(cluster3_free_times, (0, (5 - len(cluster3_free_times))),
                                             constant_values=(10), mode='constant')
                cluster4_free_times = np.pad(cluster4_free_times, (0, (5 - len(cluster4_free_times))),
                                             constant_values=(10), mode='constant')

                il_features = []

                ## Populate execution time profile (cluster-wise)
                task_exec_times = []
                for PE in self.PEs:
                    ## Skip analyzing resources if resource is MEMORY or CACHE
                    if 'MEM' in PE.name or 'CAC' in PE.name:
                        continue

                    if PE.ID == 0 or PE.ID == 4 or PE.ID == 8 or PE.ID == 10 or PE.ID == 14:
                        if shortest_task.name in self.resource_matrix.list[PE.ID].supported_functionalities:
                            ind = self.resource_matrix.list[PE.ID].supported_functionalities.index(shortest_task.name)
                            exec_time = self.resource_matrix.list[PE.ID].performance[ind]
                            task_exec_times.append(exec_time)
                        else:
                            task_exec_times.append(10000)
                        ## if shortest_task.name in self.resource_matrix.list[PE.ID].supported_functionalities :
                    ## if PE.ID == 0 or PE.ID == 4 or PE.ID == 8 or PE.ID == 10 or PE.ID == 14:
                ## for PE in self.PEs :

                ## Normalize cluster execution times
                task_exec_times = np.array(task_exec_times)
                # if len(task_exec_times[np.all([task_exec_times != 0, task_exec_times != 10000])]) == 0 :
                if len(task_exec_times[np.all([task_exec_times != 0])]) == 0:
                    normalized_task_exec_times = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
                else:
                    normalized_task_exec_times = common.get_normalized_list(task_exec_times)
                ## if len(task_exec_times[np.all([task_exec_times != 0, task_exec_times != 10000])]) == 0 :

                ## Populate normalized downward depth of task in DAG
                task_depth = shortest_task.dag_depth
                task_job_index = -1
                for job_index, job in enumerate(self.jobs.list):
                    if shortest_task.jobname == job.name:
                        job_depth = self.jobs.list[job_index].dag_depth['DAG']
                        num_tasks = len(self.jobs.list[job_index].task_list)
                        task_job_index = job_index
                normalized_task_depth = task_depth / job_depth

                ## Populate relative job ID
                if max_job_id - min_job_id == 0:
                    relative_job_id = 0
                else:
                    relative_job_id = (shortest_task.jobID - min_job_id) / (max_job_id - min_job_id)
                ## if max_job_id - min_job_id == 0 :

                pred_task_comm_time_list = []
                predecessor_PEs = []

                ## Populate predecessor PE IDs
                task_preds = np.array(shortest_task.preds)
                task_pred_IDs = task_preds + shortest_task.head_ID
                task_preds_cluster_list = []
                task_preds_comm_vol = []
                for task_pred in task_pred_IDs:
                    for completed_task in common.TaskQueues.completed.list:
                        if completed_task.ID == task_pred:
                            task_preds_cluster_list.append(common.get_cluster(completed_task.PE_ID))
                            c_vol = self.jobs.list[task_job_index].comm_vol[
                                task_pred - shortest_task.head_ID, shortest_task.base_ID]
                            task_preds_comm_vol.append(c_vol / 1000)
                            ## for PE in self.PEs :
                        ## if completed_task.ID == task_pred :
                    ## for completed_task in common.TaskQueues.completed.list :
                ## for task_pred in task_preds :

                num_short_preds = 5 - len(shortest_task.preds)
                for num in range(num_short_preds):
                    task_preds = np.append(task_preds, 10000)
                    task_preds_cluster_list.append(10000)
                    task_preds_comm_vol.append(10000)
                ## for num in range(num_short_preds) :

                task_preds_cluster_list = np.array(task_preds_cluster_list)
                task_preds_cluster_list = task_preds_cluster_list / 4
                task_preds_cluster_list[task_preds_cluster_list > 1000] = 10
                task_preds[task_preds > 1000] = 50
                task_preds_comm_vol = np.array(task_preds_comm_vol)
                task_preds_comm_vol[task_preds_comm_vol > 1000] = 10

                # Populate job type
                if shortest_task.jobname == 'WiFi_Transmitter':
                    job_type = 0
                if shortest_task.jobname == 'WiFi_Receiver':
                    job_type = 1
                if shortest_task.jobname == 'lag_detection':
                    job_type = 2
                if shortest_task.jobname == 'SC-T':
                    job_type = 3
                if shortest_task.jobname == 'SC-R':
                    job_type = 4
                if shortest_task.jobname == 'Temporal_Mitigation':
                    job_type = 5

                # Populate normalized task ID
                normalized_task_ID = shortest_task.base_ID / num_tasks

                # Populate features and save samples
                il_features.append(normalized_task_ID)
                il_features.extend(normalized_task_exec_times)
                il_features.append(normalized_task_depth)
                il_features.append(relative_job_id)
                il_features.append(job_type)
                il_features.extend(task_preds)
                il_features.extend(task_preds_cluster_list)
                il_features.extend(task_preds_comm_vol)

                ## Print resource label
                resource_label = shortest_task.PE_ID
                cluster_label = common.get_cluster(shortest_task.PE_ID)

                ## Write data onto file 
                common.ils_clustera_fp.write(str(self.env.now) + ',')
                common.ils_clustera_fp.write(str(shortest_task.ID))
                for feature in cluster_free_times:
                    common.ils_clustera_fp.write(',' + str(feature))
                for feature in il_features:
                    common.ils_clustera_fp.write(',' + str(feature))
                common.ils_clustera_fp.write(',' + str(resource_label))
                common.ils_clustera_fp.write(',' + str(cluster_label))
                common.ils_clustera_fp.write('\n')

                if cluster_label == 0:
                    common.ils_cluster0_fp.write(str(self.env.now) + ',')
                    common.ils_cluster0_fp.write(str(shortest_task.ID))
                    for feature in cluster0_free_times:
                        common.ils_cluster0_fp.write(',' + str(feature))
                    for feature in il_features:
                        common.ils_cluster0_fp.write(',' + str(feature))
                    common.ils_cluster0_fp.write(',' + str(resource_label))
                    common.ils_cluster0_fp.write(',' + str(cluster_label))
                    common.ils_cluster0_fp.write('\n')
                if cluster_label == 1:
                    common.ils_cluster1_fp.write(str(self.env.now) + ',')
                    common.ils_cluster1_fp.write(str(shortest_task.ID))
                    for feature in cluster1_free_times:
                        common.ils_cluster1_fp.write(',' + str(feature))
                    for feature in il_features:
                        common.ils_cluster1_fp.write(',' + str(feature))
                    common.ils_cluster1_fp.write(',' + str(resource_label))
                    common.ils_cluster1_fp.write(',' + str(cluster_label))
                    common.ils_cluster1_fp.write('\n')
                if cluster_label == 2:
                    common.ils_cluster2_fp.write(str(self.env.now) + ',')
                    common.ils_cluster2_fp.write(str(shortest_task.ID))
                    for feature in cluster2_free_times:
                        common.ils_cluster2_fp.write(',' + str(feature))
                    for feature in il_features:
                        common.ils_cluster2_fp.write(',' + str(feature))
                    common.ils_cluster2_fp.write(',' + str(resource_label))
                    common.ils_cluster2_fp.write(',' + str(cluster_label))
                    common.ils_cluster2_fp.write('\n')
                if cluster_label == 3:
                    common.ils_cluster3_fp.write(str(self.env.now) + ',')
                    common.ils_cluster3_fp.write(str(shortest_task.ID))
                    for feature in cluster3_free_times:
                        common.ils_cluster3_fp.write(',' + str(feature))
                    for feature in il_features:
                        common.ils_cluster3_fp.write(',' + str(feature))
                    common.ils_cluster3_fp.write(',' + str(resource_label))
                    common.ils_cluster3_fp.write(',' + str(cluster_label))
                    common.ils_cluster3_fp.write('\n')
                if cluster_label == 4:
                    common.ils_cluster4_fp.write(str(self.env.now) + ',')
                    common.ils_cluster4_fp.write(str(shortest_task.ID))
                    for feature in cluster4_free_times:
                        common.ils_cluster4_fp.write(',' + str(feature))
                    for feature in il_features:
                        common.ils_cluster4_fp.write(',' + str(feature))
                    common.ils_cluster4_fp.write(',' + str(resource_label))
                    common.ils_cluster4_fp.write(',' + str(cluster_label))
                    common.ils_cluster4_fp.write('\n')


            ##################################################################################################
            ## End of IL-Scheduler
            ##################################################################################################

            # Finally, update the estimated available time of the resource to which
            # a task is just assigned
            self.PEs[shortest_task.PE_ID].available_time = self.env.now + shortest_comparison[shortest_task.PE_ID]

            # Remove the task which got a schedule successfully
            for i, task in enumerate(ready_list):
                if task.ID == shortest_task.ID:
                    ready_list.remove(task)

            task_counter += 1
            # At the end of this loop, we should have a valid (non-negative ID)
            # that can run next_task

        # end of while len(ready_list) > 0 :

        ## Close file handles
        if common.ils_enable_dataset_save:
            common.ils_close_file_handles()
        ## if common.ils_enable_dataset_save :

    # end of ETF( list_of_ready)

    def CP_CLUSTER(self, list_of_ready):
        '''
        This scheduler finds a schedule using CP formulation
        '''
        if common.job_name_temp == 'WiFi_Transmitter':
            deadline = 150
        elif common.job_name_temp == 'WiFi_Receiver':
            deadline = 400
        elif common.job_name_temp == 'SC-R':
            deadline = 300
        elif common.job_name_temp == 'SC-T':
            deadline = 150
        elif common.job_name_temp == 'Temporal_Mitigation':
            deadline = 160
        elif common.job_name_temp == 'Top':
            deadline = 100
        elif common.job_name_temp == 'pulse_doppler':
            deadline = 1000
        elif common.job_name_temp == 'lag_detection':
            deadline = 300
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = self.env.now + deadline
        for task in list_of_ready:
            PE_ID_list = []

            if (common.DEBUG_SCH):
                print('[D] Time %s: The scheduler function is called with task %s'
                      % (self.env.now, task.ID))

            for cluster_schedule in common.temp_list:
                for schedule in cluster_schedule:
                    if task.ID == schedule[0]:
                        # print(self.env.now, task.ID, schedule)
                        for i, PE in enumerate(self.resource_matrix.list):
                            if PE.type == schedule[1]:
                                PE_ID_list.append(PE.ID)
                                # PE_available_list.append(self.PEs[PE.ID].available_time)

                        comparison = [np.inf] * len(PE_ID_list)
                        comm_ready = [0] * len(PE_ID_list)

                        for i in range(len(self.resource_matrix.list)):
                            # if the task is supported by the resource, retrieve the index of the task
                            # if (task.name in self.resource_matrix.list[i].supported_functionalities):
                            if self.resource_matrix.list[i].ID in PE_ID_list:
                                PE_ind = PE_ID_list.index(i)
                                ind = self.resource_matrix.list[i].supported_functionalities.index(task.name)

                                # $PE_comm_wait_times is a list to store the estimated communication time
                                # (or the remaining communication time) of all predecessors of a task for a PE
                                # As simulation forwards, relevant data is being sent after a task is completed
                                # based on the time instance, one should consider either whole communication
                                # time or the remaining communication time for scheduling
                                PE_comm_wait_times = []

                                # $PE_wait_time is a list to store the estimated wait times for a PE
                                # till that PE is available if the PE is currently running a task
                                PE_wait_time = []

                                job_ID = -1  # Initialize the job ID

                                # Retrieve the job ID which the current task belongs to
                                for ii, job in enumerate(self.jobs.list):
                                    if job.name == task.jobname:
                                        job_ID = ii

                                for predecessor in self.jobs.list[job_ID].task_list[task.base_ID].predecessors:
                                    # data required from the predecessor for $ready_task
                                    c_vol = self.jobs.list[job_ID].comm_vol[predecessor, task.base_ID]

                                    # retrieve the real ID  of the predecessor based on the job ID
                                    real_predecessor_ID = predecessor + task.ID - task.base_ID

                                    # Initialize following two variables which will be used if
                                    # PE to PE communication is utilized
                                    predecessor_PE_ID = -1
                                    predecessor_finish_time = -1

                                    for completed in common.TaskQueues.completed.list:
                                        if completed.ID == real_predecessor_ID:
                                            predecessor_PE_ID = completed.PE_ID
                                            predecessor_finish_time = completed.finish_time
                                            # print(predecessor, predecessor_finish_time, predecessor_PE_ID)

                                    if (common.PE_to_PE):
                                        # Compute the PE to PE communication time
                                        PE_to_PE_band = common.ResourceManager.comm_band[predecessor_PE_ID, i]
                                        PE_to_PE_comm_time = int(c_vol / PE_to_PE_band)

                                        PE_comm_wait_times.append(
                                            max((predecessor_finish_time + PE_to_PE_comm_time - self.env.now), 0))

                                        if (common.DEBUG_SCH):
                                            print(
                                                '[D] Time %s: Estimated communication time between PE %s to PE %s from task %s to task %s is %d'
                                                % (self.env.now, predecessor_PE_ID, i, real_predecessor_ID, task.ID,
                                                   PE_comm_wait_times[-1]))

                                    if (common.shared_memory):
                                        # Compute the communication time considering the shared memory
                                        # only consider memory to PE communication time
                                        # since the task passed the 1st phase (PE to memory communication)
                                        # and its status changed to ready

                                        # PE_to_memory_band = self.resource_matrix.comm_band[predecessor_PE_ID, -1]
                                        memory_to_PE_band = self.resource_matrix.comm_band[
                                            self.resource_matrix.list[-1].ID, i]
                                        shared_memory_comm_time = int(c_vol / memory_to_PE_band)

                                        PE_comm_wait_times.append(shared_memory_comm_time)
                                        if (common.DEBUG_SCH):
                                            print(
                                                '[D] Time %s: Estimated communication time between memory to PE %s from task %s to task %s is %d'
                                                % (
                                                    self.env.now, i, real_predecessor_ID, task.ID,
                                                    PE_comm_wait_times[-1]))

                                    # $comm_ready contains the estimated communication time
                                    # for the resource in consideration for scheduling
                                    # maximum value is chosen since it represents the time required for all
                                    # data becomes available for the resource.
                                    comm_ready[PE_ind] = (max(PE_comm_wait_times))
                                # end of for for predecessor in self.jobs.list[job_ID].task_list[ind].predecessors:

                                # if a resource currently is executing a task, then the estimated remaining time
                                # for the task completion should be considered during scheduling
                                PE_wait_time.append(max((self.PEs[i].available_time - self.env.now), 0))

                                # update the comparison vector accordingly
                                comparison[PE_ind] = self.resource_matrix.list[i].performance[ind] + max(
                                    comm_ready[PE_ind], PE_wait_time[-1])
                            # if self.resource_matrix.list[i].ID in PE_ID_list:
                        # end of for i in range(len(self.resource_matrix.list)):

                        # after going over each resource, choose the one which gives the minimum result
                        min_ind = comparison.index(min(comparison))
                        task.PE_ID = PE_ID_list[min_ind]

                        if task.PE_ID == -1:
                            print(
                                '[E] Time %s: %s can not be assigned to any resource, please check DASH.SoC.**.txt file'
                                % (self.env.now, task.ID))
                            print('[E] or job_**.txt file')
                            assert (task.PE_ID >= 0)
                        else:
                            if (common.DEBUG_SCH):
                                print('[D] Time %s: Estimated execution times for each PE with task %s, respectively'
                                      % (self.env.now, task.ID))
                                print('%12s' % (''), comparison)
                                print('[D] Time %s: The scheduler assigns task %s to resource %s: %s'
                                      % (self.env.now, task.ID, task.PE_ID, self.resource_matrix.list[task.PE_ID].name))

                        # Finally, update the estimated available time of the resource to which
                        # a task is just assigned
                        self.PEs[task.PE_ID].available_time = self.env.now + comparison[PE_ind]

                        # At the end of this loop, we should have a valid (non-negative ID)
                        # that can run next_task

        # end of for task in list_of_ready:

    def CP(self, list_of_ready):
        '''!
        This scheduler utilizes a look-up table for scheduling tasks to a particular processor
        @param list_of_ready: The list of ready tasks
        '''
        for task in list_of_ready:
            ind = 0
            base = 0
            for item in common.ilp_job_list:
                if item[0] == task.jobID:
                    ind = common.ilp_job_list.index(item)
                    break

            previous_job_list = list(range(ind))
            for job in previous_job_list:
                selection = common.ilp_job_list[job][1]
                num_of_tasks = len(self.jobs.list[selection].task_list)
                base += num_of_tasks

            # print(task.jobID, base, task.base_ID)

            for i, schedule in enumerate(common.table):

                if len(common.table) > base:
                    if (task.base_ID + base) == i:
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]
                else:
                    if (task.ID % num_of_tasks == i):
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]

        list_of_ready.sort(key=lambda x: x.order, reverse=False)

    # end of CP_Cluster(list_of_ready)

    def CP_PE(self, list_of_ready):
        '''
        This scheduler utilizes a look-up table for scheduling tasks
        to a particular processor
        '''
        if common.job_name_temp == 'WiFi_Transmitter':
            deadline = 150
        elif common.job_name_temp == 'WiFi_Receiver':
            deadline = 400
        elif common.job_name_temp == 'SC-R':
            deadline = 300
        elif common.job_name_temp == 'SC-T':
            deadline = 150
        elif common.job_name_temp == 'Temporal_Mitigation':
            deadline = 160
        elif common.job_name_temp == 'Top':
            deadline = 100
        elif common.job_name_temp == 'pulse_doppler':
            deadline = 1000
        elif common.job_name_temp == 'lag_detection':
            deadline = 300
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = self.env.now + deadline
        for task in list_of_ready:
            for i, schedule in enumerate(common.table):
                num_of_tasks = len(self.jobs.list[0].task_list)
                ind = common.ilp_job_list.index(task.jobID)

                if len(common.table) > num_of_tasks:
                    if (task.base_ID + ind * num_of_tasks) == i:
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]
                else:
                    if (task.ID % num_of_tasks == i):
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]

        list_of_ready.sort(key=lambda x: x.order, reverse=False)
        # def CP_PE(self, list_of_ready):

    def CP_MULTI(self, list_of_ready):
        '''
        This scheduler utilizes a look-up table for scheduling tasks
        to a particular processor
        '''
        if common.job_name_temp == 'WiFi_Transmitter':
            deadline = 150
        elif common.job_name_temp == 'WiFi_Receiver':
            deadline = 400
        elif common.job_name_temp == 'SC-R':
            deadline = 300
        elif common.job_name_temp == 'SC-T':
            deadline = 150
        elif common.job_name_temp == 'Temporal_Mitigation':
            deadline = 160
        elif common.job_name_temp == 'Top':
            deadline = 100
        elif common.job_name_temp == 'pulse_doppler':
            deadline = 1000
        elif common.job_name_temp == 'lag_detection':
            deadline = 300
        for task in list_of_ready:
            if task.jobID not in common.deadline_dict:
                common.deadline_dict[task.jobID] = self.env.now + deadline
        for task in list_of_ready:
            ind = 0
            base = 0
            for item in common.ilp_job_list:
                if item[0] == task.jobID:
                    ind = common.ilp_job_list.index(item)
                    break

            previous_job_list = list(range(ind))
            for job in previous_job_list:
                selection = common.ilp_job_list[job][1]
                num_of_tasks = len(self.jobs.list[selection].task_list)
                base += num_of_tasks

            # print(task.jobID, base, task.base_ID)

            for i, schedule in enumerate(common.table):

                if len(common.table) > base:
                    if (task.base_ID + base) == i:
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]
                else:
                    if (task.ID % num_of_tasks == i):
                        task.PE_ID = schedule[0]
                        task.order = schedule[1]

        list_of_ready.sort(key=lambda x: x.order, reverse=False)
        # def CP_MULTI(self, list_of_ready):
