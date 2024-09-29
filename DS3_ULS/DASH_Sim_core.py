'''
Description: This file contains the simulation core that handles the simulation events.
'''
import sys
import os
import csv

import DTPM_power_models
import common  # The common parameters used in DASH-Sim are defined in common_parameters.py
import DTPM
import DASH_Sim_utils
import DTPM_policies


# Define the core of the simulation engine
# This function calls the scheduler, starts/interrupts the tasks,
# and manages collection of all the statistics

class SimulationManager:
    '''
    Define the SimulationManager class to handle the simulation events.
    '''

    def __init__(self, env, sim_done, job_gen, scheduler, PE_list, jobs, resource_matrix):
        '''
        env: Pointer to the current simulation environment
        scheduler: Pointer to the DASH_scheduler
        PE_list: The PEs available in the current SoC
        jobs: The list of all jobs given to DASH-Sim
        resource_matrix: The data structure that defines power/performance
            characteristics of the PEs for each supported task
        '''
        self.env = env
        self.sim_done = sim_done
        self.job_gen = job_gen
        self.scheduler = scheduler
        self.PEs = PE_list
        self.jobs = jobs
        self.resource_matrix = resource_matrix

        # 调用调度算法
        self.action = env.process(self.run())  # starts the run() method as a SimPy process

    # As the simulation proceeds, tasks are being processed.
    # We need to update the ready tasks queue after completion of each task
    # 在processing_element.run中调用
    def update_ready_queue(self, completed_task):
        '''
        This function updates the common.TaskQueues.ready after one task is completed.
        '''

        # completed_task is the task whose processing is just completed
        # Add completed task to the completed tasks queue
        # 更新completed.list
        common.TaskQueues.completed.list.append(completed_task)
        common.TaskQueues.completed.set.add(completed_task.ID)

        # Remove the completed task from the queue of the PE
        # 从PE执行队列中移除completed_task
        for task in self.PEs[completed_task.PE_ID].queue:
            if task.ID == completed_task.ID:
                self.PEs[task.PE_ID].queue.remove(task)

        # Remove the completed task from the currently running queue
        # 从running.list中移除completed_task
        common.TaskQueues.running.list.remove(completed_task)

        # Remove the completed task from the current DAG representation
        # 从当前dag中移除completed_task对应的node
        if completed_task.ID in common.current_dag:
            common.current_dag.remove_node(completed_task.ID)

        # Initialize $remove_from_outstanding_queue which will populate tasks
        # to be removed from the outstanding queue
        remove_from_outstanding_queue = []

        # Initialize $to_memory_comm_time which will be communication time to
        # memory for data from a predecessor task to a outstanding task
        to_memory_comm_time = -1

        job_ID = -1
        # 找到当前task所在的job_id
        for ind, job in enumerate(self.jobs.list):
            if job.name == completed_task.jobname:
                job_ID = ind

        # Check if the dependency of any outstanding task is cleared
        # We need to move them to the ready queue
        # 判断在该task完成后，是否有依赖关系的更新，即部分task从outstanding变为ready
        # 遍历outstanding队列
        for i, outstanding_task in enumerate(common.TaskQueues.outstanding.list):  # Go over each outstanding task
            for ii, predecessor in enumerate(outstanding_task.predecessors):  # Go over each predecessor
                if completed_task.ID in outstanding_task.predecessors:  # if the completed task is one of the predecessors
                    outstanding_task.predecessors.remove(completed_task.ID)  # Clear this predecessor
                    #  不进
                    if (common.shared_memory):
                        # Get the communication time to memory for data from a 
                        # predecessor task to a outstanding task 
                        comm_vol = self.jobs.list[job_ID].comm_vol[completed_task.base_ID, outstanding_task.base_ID]
                        comm_band = common.ResourceManager.comm_band[
                            completed_task.PE_ID, self.resource_matrix.list[-1].ID]
                        to_memory_comm_time = int(comm_vol / comm_band)  # Communication time from a PE to memory

                        if (common.DEBUG_SIM):
                            print('[D] Time %d: Data from task %d for task %d will be sent to memory in %d us'
                                  % (self.env.now, completed_task.ID, outstanding_task.ID, to_memory_comm_time))

                        # Based on this communication time, this outstanding task
                        # will be added to the ready queue. That is why, keep track of
                        # all communication times required for a task in the list
                        # $ready_wait_times
                        outstanding_task.ready_wait_times.append(to_memory_comm_time + self.env.now)
                    # end of if (common.shared_memory):

                # end of if (completed_task.ID in outstanding_task.predecessors):
            # end of for ii, predecessor in enumerate(outstanding_task.predecessors):

            # 如果该节点没有前驱节点
            no_predecessors = (len(outstanding_task.predecessors) == 0)  # Check if this was the last dependency
            # 如果该节点在running_queue中
            currently_running = (outstanding_task in  # if the task is in the running queue,
                                 common.TaskQueues.running.list)  # We should not put it back to the ready queue
            # 如果该节点不在ready.list中
            not_in_ready_queue = not (outstanding_task in  # If this task is already in the ready queue,
                                      common.TaskQueues.ready.list)  # We should not append another copy

            # 如果该节点没有前驱节点,不在running_queue中,不在ready.list中
            if (no_predecessors and not (currently_running) and not_in_ready_queue):
                if (common.PE_to_PE):  # if PE to PE communication is utilized
                    # ready.list添加该节点
                    common.TaskQueues.ready.list.append(
                        common.TaskQueues.outstanding.list[i])  # Add the task to the ready queue immediately
                # 不进
                elif (common.shared_memory):
                    # if shared memory is utilized for communication, then
                    # the outstanding task will wait for a certain amount time
                    # (till the $time_stamp)for being added into the ready queue
                    common.TaskQueues.wait_ready.list.append(outstanding_task)
                    if (common.INFO_SIM) and (common.shared_memory):
                        print('[I] Time %d: Task %d ready times due to memory communication of its predecessors are'
                              % (self.env.now, outstanding_task.ID))
                        print('%12s' % (''), outstanding_task.ready_wait_times)
                    common.TaskQueues.wait_ready.list[-1].time_stamp = max(outstanding_task.ready_wait_times)

                # 删除该节点
                remove_from_outstanding_queue.append(outstanding_task)
        # end of for i, outstanding_task in...

        # Remove the tasks from outstanding queue that have been moved to ready queue
        for task in remove_from_outstanding_queue:
            common.TaskQueues.outstanding.list.remove(task)

        # At the end of this function:
        # Newly processed $completed_task is added to the completed tasks
        # outstanding tasks with no dependencies are added to the ready queue
        # based on the communication mode and then, they are removed from
        # the outstanding queue

    # end def update_ready_queue(completed_task)

    # 在DASH_Sim_core.run中调用
    def update_execution_queue(self, ready_list):
        '''
        This function updates the common.TaskQueues.executable if one task is ready
        for execution but waiting for the communication time, either between
        memory and a PE, or between two PEs (based on the communication mode)
        '''
        # Initialize $remove_from_ready_queue which will populate tasks
        # to be removed from the outstanding queue
        remove_from_ready_queue = []

        # Initialize $from_memory_comm_time which will be communication time 
        # for data from memory to a PE
        from_memory_comm_time = -1

        # Initialize $PE_to_PE_comm_time which will be communication time
        # for data from a PE to another PE
        PE_to_PE_comm_time = -1

        job_ID = -1
        # 遍历ready_list中的task
        i = 0
        for ready_task in ready_list:
            # 对每个task都通过遍历找到其所在的job_id
            for ind, job in enumerate(self.jobs.list):
                if job.name == ready_task.jobname:
                    job_ID = ind

            # 遍历ready_list中task所在job的所有task
            for i, task in enumerate(self.jobs.list[job_ID].task_list):
                # 找到job的task中对应ready_task的那个
                if ready_task.base_ID == task.ID:
                    # 如果该task为起始节点
                    if ready_task.head:
                        # if a task is the leading task of a job
                        # then it can start immediately since it has no predecessor
                        ready_task.PE_to_PE_wait_time.append(self.env.now)
                        ready_task.execution_wait_times.append(self.env.now)
                    # end of if ready_task.head == True:

                    # 遍历该task的前驱节点数组
                    for predecessor in task.predecessors:
                        # 找到相对应的节点
                        if task.ID == ready_task.ID:
                            ready_task.predecessors = task.predecessors
                        # data required from the predecessor for $ready_task
                        comm_vol = self.jobs.list[job_ID].comm_vol[predecessor, ready_task.base_ID]

                        # retrieve the real ID  of the predecessor based on the job ID
                        real_predecessor_ID = predecessor + ready_task.ID - ready_task.base_ID

                        # Initialize following two variables which will be used if 
                        # PE to PE communication is utilized
                        predecessor_PE_ID = -1
                        predecessor_finish_time = -1

                        # 此时进入该分支
                        if common.PE_to_PE:
                            # Compute the PE to PE communication time
                            for completed in common.TaskQueues.completed.list:
                                if completed.ID == real_predecessor_ID:
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time
                            comm_band = common.ResourceManager.comm_band[predecessor_PE_ID, ready_task.PE_ID]
                            if predecessor_PE_ID == ready_task.PE_ID:
                                PE_to_PE_comm_time = 0
                            else:
                                # print(ready_task.ID, predecessor_PE_ID, ready_task.PE_ID, comm_vol, comm_band)
                                PE_to_PE_comm_time = int(comm_vol / comm_band)
                            # PE等待时间为前驱节点结束时间+PE到PE通信时间，从所有前驱节点计算出的等待时间放入数组中
                            # if ready_task.isChange:
                            #     ready_task.PE_to_PE_wait_time.append(PE_to_PE_comm_time + self.env.now)
                            #     ready_task.isChange = False
                            # else:
                            ready_task.PE_to_PE_wait_time.append(PE_to_PE_comm_time + predecessor_finish_time)

                            if common.DEBUG_SIM:
                                print(
                                    '[D] Time %d: Data transfer from PE-%s to PE-%s for task %d from task %d is completed at %d us'
                                    % (self.env.now, predecessor_PE_ID, ready_task.PE_ID,
                                       ready_task.ID, real_predecessor_ID, ready_task.PE_to_PE_wait_time[-1]))
                        # end of if (common.PE_to_PE): 

                        # 不进入该分支
                        if (common.shared_memory):
                            # Compute the memory to PE communication time
                            comm_band = common.ResourceManager.comm_band[
                                self.resource_matrix.list[-1].ID, ready_task.PE_ID]
                            from_memory_comm_time = int(comm_vol / comm_band)
                            if (common.DEBUG_SIM):
                                print(
                                    '[D] Time %d: Data from memory for task %d from task %d will be sent to PE-%s in %d us'
                                    % (self.env.now, ready_task.ID, real_predecessor_ID, ready_task.PE_ID,
                                       from_memory_comm_time))
                            ready_task.execution_wait_times.append(from_memory_comm_time + self.env.now)
                        # end of if (common.shared_memory)
                    # end of for predecessor in task.predecessors:

                    # Populate all ready tasks in executable with a time stamp
                    # which will show when a task is ready for execution
                    common.TaskQueues.executable.list.append(ready_task)
                    remove_from_ready_queue.append(ready_task)
                    # 取数组中最大的等待时间
                    if common.PE_to_PE:
                        common.TaskQueues.executable.list[-1].time_stamp = max(ready_task.PE_to_PE_wait_time)
                    else:
                        common.TaskQueues.executable.list[-1].time_stamp = max(ready_task.execution_wait_times)

                # end of ready_task.base_ID == task.ID:
            # end of i, task in enumerate(self.jobs.list[job_ID].task_list):    
        # end of for ready_task in ready_list:

        # Remove the tasks from ready queue that have been moved to executable queue
        for task in remove_from_ready_queue:
            common.TaskQueues.ready.list.remove(task)

        if self.scheduler.name != 'OBO' and self.scheduler.name != 'HEFT_RT' and self.scheduler.name != 'PEFT' and self.scheduler.name != 'PEFT_RT' and self.scheduler.name != "ProLis" and self.scheduler.name != "LookAhead" and self.scheduler.name != "CostEfficient" and self.scheduler.name != "CostEfficient" and self.scheduler.name != "ULS" and self.scheduler.name != "ALAP_RT_EDP" and self.scheduler.name != "SDP_EC" and self.scheduler.name != "Prob" and self.scheduler.name != "dynProb":
            # Reorder tasks based on their job IDs
            common.TaskQueues.executable.list.sort(key=lambda task: task.jobID, reverse=False)

    # 在DASH_Sim_core.run中调用
    def update_execution_queue_1(self, executable_list):
        '''
        This function updates the common.TaskQueues.executable if one task is ready
        for execution but waiting for the communication time, either between
        memory and a PE, or between two PEs (based on the communication mode)
        '''
        job_ID = -1
        # 遍历ready_list中的task
        i = 0
        for executable_task in executable_list:
            # 对每个task都通过遍历找到其所在的job_id
            for ind, job in enumerate(self.jobs.list):
                if job.name == executable_task.jobname:
                    job_ID = ind

            # 遍历ready_list中task所在job的所有task
            for i, task in enumerate(self.jobs.list[job_ID].task_list):
                # 找到job的task中对应ready_task的那个
                if executable_task.base_ID == task.ID:
                    # 如果该task为起始节点
                    if executable_task.head:
                        # if a task is the leading task of a job
                        # then it can start immediately since it has no predecessor
                        executable_task.PE_to_PE_wait_time.append(self.env.now)
                        executable_task.execution_wait_times.append(self.env.now)
                    # end of if ready_task.head == True:

                    # 遍历该task的前驱节点数组
                    for predecessor in task.predecessors:
                        # 找到相对应的节点
                        if task.ID == executable_task.ID:
                            executable_task.predecessors = task.predecessors
                        # data required from the predecessor for $ready_task
                        comm_vol = self.jobs.list[job_ID].comm_vol[predecessor, executable_task.base_ID]

                        # retrieve the real ID  of the predecessor based on the job ID
                        real_predecessor_ID = predecessor + executable_task.ID - executable_task.base_ID

                        # Initialize following two variables which will be used if
                        # PE to PE communication is utilized
                        predecessor_PE_ID = -1
                        predecessor_finish_time = -1

                        # 此时进入该分支
                        if common.PE_to_PE:
                            completed_1 = -1
                            # Compute the PE to PE communication time
                            for completed in common.TaskQueues.completed.list:
                                if completed.ID == real_predecessor_ID:
                                    completed_1 = completed
                                    predecessor_PE_ID = completed.PE_ID
                                    predecessor_finish_time = completed.finish_time

                            comm_band = common.ResourceManager.comm_band[predecessor_PE_ID, executable_task.PE_ID]
                            if predecessor_PE_ID == executable_task.PE_ID:
                                PE_to_PE_comm_time = 0
                            else:
                                # print(ready_task.ID, predecessor_PE_ID, ready_task.PE_ID, comm_vol, comm_band)
                                PE_to_PE_comm_time = int(comm_vol / comm_band)
                            # PE等待时间为前驱节点结束时间+PE到PE通信时间，从所有前驱节点计算出的等待时间放入数组中
                            if completed_1 != -1 and executable_task.isChange:
                                executable_task.PE_to_PE_wait_time.append(PE_to_PE_comm_time + self.env.now)
                                executable_task.isChange = False
                            else:
                                executable_task.PE_to_PE_wait_time.append(PE_to_PE_comm_time + predecessor_finish_time)
                                executable_task.isChange = False

                            if common.DEBUG_SIM:
                                print(
                                    '[D] Time %d: Data transfer from PE-%s to PE-%s for task %d from task %d is completed at %d us'
                                    % (self.env.now, predecessor_PE_ID, executable_task.PE_ID,
                                       executable_task.ID, real_predecessor_ID, executable_task.PE_to_PE_wait_time[-1]))
                        # end of if (common.PE_to_PE):

                    # 取数组中最大的等待时间
                    if common.PE_to_PE:
                        #     for task in common.TaskQueues.executable.list:
                        #         if task.ID == executable_task:
                        # print(executable_task.time_stamp)
                        executable_task.time_stamp = max(executable_task.PE_to_PE_wait_time)
                        # print(executable_task.time_stamp)
                        # print()
                        # common.TaskQueues.executable.list[-1].time_stamp = max(executable_task.PE_to_PE_wait_time)
                    else:
                        common.TaskQueues.executable.list[-1].time_stamp = max(executable_task.execution_wait_times)

            # end of ready_task.base_ID == task.ID:
            # end of i, task in enumerate(self.jobs.list[job_ID].task_list):
        # end of for ready_task in ready_list:

    # 在processing_element.run中调用
    def update_completed_queue(self):
        '''
        This function updates the common.TaskQueues.completed 
        '''
        ## Be careful about this function when there are diff jobs in the system
        # reorder tasks based on their job IDs
        common.TaskQueues.completed.list.sort(key=lambda x: x.jobID, reverse=False)

        # first_task_jobID = common.TaskQueues.completed.list[0].jobID
        # last_task_jobID = common.TaskQueues.completed.list[-1].jobID

        # if ((last_task_jobID - first_task_jobID) > 15):
        #     for i, task in enumerate(common.TaskQueues.completed.list):
        #         if (task.jobID == first_task_jobID):
        #             del common.TaskQueues.completed.list[i]

    # Implement the basic run method that will be called periodically
    # in each simulation "tick"
    def run(self):
        '''
        This function takes the next ready tasks and run on the specific PE 
        and update the common.TaskQueues.ready list accordingly.
        '''
        DTPM_module = DTPM.DTPMmodule(self.env, self.resource_matrix, self.PEs)

        for cluster in common.ClusterManager.cluster_list:
            DTPM_policies.initialize_frequency(cluster)
            # print('\n'.join(['%s:%s' % item for item in cluster.__dict__.items()]))
            # print()
        while True:  # Continue till the end of the simulation
            if self.env.now % common.sampling_rate == 0:
                # common.results.job_counter_list.append(common.results.job_counter)
                # common.results.sampling_rate_list.append(self.env.now)
                # Evaluate idle PEs, busy PEs will be updated and evaluated from the PE class
                # if self.scheduler.name != 'SDP_EC' and self.scheduler.name != 'HEFT_EDP_LB':
                DTPM_module.evaluate_idle_PEs()
            # end of if self.env.now % common.sampling_rate == 0:
            # 现在没用
            if common.shared_memory:
                # this section is activated only if shared memory is used

                # Initialize $remove_from_wait_ready which will populate tasks
                # to be removed from the wait ready queue
                remove_from_wait_ready = []

                for i, waiting_task in enumerate(common.TaskQueues.wait_ready.list):
                    if waiting_task.time_stamp <= self.env.now:
                        common.TaskQueues.ready.list.append(waiting_task)
                        remove_from_wait_ready.append(waiting_task)
                # at the end of this loop, all the waiting tasks with a time stamp
                # equal or smaller than the simulation time will be added to
                # the ready queue list
                # end of for i, waiting_task in...

                # Remove the tasks from wait ready queue that have been moved to ready queue
                for task in remove_from_wait_ready:
                    common.TaskQueues.wait_ready.list.remove(task)
            # end of if (common.shared_memory):
            # 现在没用
            if (common.INFO_SIM) and len(common.TaskQueues.ready.list) > 0:
                print('[I] Time %s: DASH-Sim ticks with %d task ready for being assigned to a PE'
                      % (self.env.now, len(common.TaskQueues.ready.list)))
            # for i in self.PEs:
            #     print(i.name, DTPM_power_models.compute_dynamic_power_dissipation(
            #             common.ClusterManager.cluster_list[i.cluster_ID].current_frequency,
            #             common.ClusterManager.cluster_list[i.cluster_ID].current_voltage,
            #             i.Cdyn_alpha))
            # print()
            # common.TaskQueues.ready.list在job_generator中生成
            if not len(common.TaskQueues.ready.list) == 0:
                # give all tasks in ready_list to the chosen scheduler
                # and scheduler will assign the tasks to a PE
                if self.scheduler.name == 'CPU_only':
                    self.scheduler.CPU_only(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'MET':
                    self.scheduler.MET(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'EFT':
                    self.scheduler.EFT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'STF':
                    self.scheduler.STF(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'DPDS':
                    self.scheduler.DPDS(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'ILS_ETF':
                    self.scheduler.ILS_ETF(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'ETF_LB':
                    self.scheduler.ETF_LB(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'OBO':
                    self.scheduler.OBO(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'HEFT_RT':
                    self.scheduler.HEFT_RT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'HEFT_EDP':
                    self.scheduler.HEFT_EDP(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'HEFT_EDP_LB':
                    self.scheduler.HEFT_EDP_LB(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'PEFT':
                    self.scheduler.PEFT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'PEFT_RT':
                    self.scheduler.PEFT_RT(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'CP_PE':
                    self.scheduler.CP_PE(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'CP_CLUSTER':
                    self.scheduler.CP_CLUSTER(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'CP_MULTI':
                    self.scheduler.CP_MULTI(common.TaskQueues.ready.list)
                elif self.scheduler.name == 'ULS':
                    self.scheduler.ULS(common.TaskQueues.ready.list)
                else:
                    print('[E] Could not find the requested scheduler')
                    print('[E] Please check "config_file.ini" and enter a proper name')
                    print('[E] or check "scheduler.py" if the scheduler exist')
                    sys.exit()
                # end of if self.scheduler.name

                # 根据ready_queue生成executable_queue
                self.update_execution_queue(common.TaskQueues.ready.list)  # Update the execution queue based on task's info
                if self.scheduler.name == 'dynProb':
                    common.TaskQueues.ready.list.sort(key=lambda task: task.order)
                    common.TaskQueues.outstanding.list.sort(key=lambda task: task.order)
                    common.TaskQueues.executable.list.sort(key=lambda task: task.order)
                # data loss
                if self.scheduler.name == 'ULS' or self.scheduler.name == 'SDP_EC':
                    self.update_execution_queue_1(common.TaskQueues.executable.list)
                    common.TaskQueues.executable.list.sort(key=lambda task: task.order)

            # end of if not len(common.TaskQueues.ready.list) == 0:
            # Initialize $remove_from_executable which will populate tasks
            # to be removed from the executable queue
            remove_from_executable = []
            # Go over each task in the executable queue
            if len(common.TaskQueues.executable.list) is not 0:
                for i, executable_task in enumerate(common.TaskQueues.executable.list):
                    # 如果当前时间大于等于可执行task的最早执行时间为true，否则为false
                    is_time_to_execute = (executable_task.time_stamp <= self.env.now)
                    # 如果对应PE的容量大于PE的执行task数为true，否则为false
                    PE_has_capacity = (
                            len(self.PEs[executable_task.PE_ID].queue) < self.PEs[executable_task.PE_ID].capacity)
                    # 如果对应可执行task已分配PE为true，否则为false
                    task_has_assignment = (executable_task.PE_ID != -1)

                    dynamic_dependencies_met = True

                    # set.intersection取交集，在此取common.TaskQueues.completed与executable_task.dynamic_dependencies的交集
                    dependencies_completed = common.TaskQueues.completed.set.intersection(
                        executable_task.dynamic_dependencies)
                    # print(dependencies_completed)

                    # 如果executable_task.dynamic_dependencies不是common.TaskQueues.completed的子集，即不是所有可执行的都满足依赖关系
                    if len(dependencies_completed) != len(executable_task.dynamic_dependencies):
                        dynamic_dependencies_met = False

                    # 满足执行的条件
                    if is_time_to_execute and PE_has_capacity and dynamic_dependencies_met and task_has_assignment:
                        # 给对应PE的执行队列挂上该task
                        self.PEs[executable_task.PE_ID].queue.append(executable_task)

                        if common.INFO_SIM:
                            print('[I] Time %s: Task %s is ready for execution by PE-%s'
                                  % (self.env.now, executable_task.ID, executable_task.PE_ID))

                        # 当前PE
                        current_resource = self.resource_matrix.list[executable_task.PE_ID]
                        # print('\n'.join(['%s:%s' % item for item in executable_task.__dict__.items()]))
                        # print()
                        # executable_task变为running_task
                        self.env.process(self.PEs[executable_task.PE_ID].run(
                            # Send the current task and a handle for this simulation manager (self)
                            self, executable_task, current_resource,
                            DTPM_module))  # This handle is used by the PE to call the update_ready_queue function

                        remove_from_executable.append(executable_task)
                    # end of if is_time_to_execute and PE_has_capacity and dynamic_dependencies_met
                # end of for i, executable_task in...
            # end of if not len(common.TaskQueues.executable.list) == 0:

            # Remove the tasks from executable queue that have been executed by a resource
            for task in remove_from_executable:
                common.TaskQueues.executable.list.remove(task)

            # If DRL scheduler is active, tha tasks waiting in the exectuable queue will be redirected to the ready queue
            if len(common.TaskQueues.executable.list):
                if self.scheduler.name == 'DRL':
                    # print('ILP' in self.scheduler.name)
                    while len(common.TaskQueues.executable.list) > 0:
                        task = common.TaskQueues.executable.list.pop(-1)
                        common.TaskQueues.ready.list.append(task)

            # The simulation tick is completed. Wait till the next interval
            yield self.env.timeout(common.simulation_clk)

            # 如果当前时间大于仿真执行时间
            if self.env.now > common.simulation_length:
                self.sim_done.succeed()
        # end while (True)
