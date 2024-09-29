'''
Description: This file contains the process elements and their attributes.
'''
import os
import sys

import simpy
import copy

import common  # The common parameters used in DASH-Sim are defined in common_parameters.py
import DTPM_power_models
import DASH_Sim_utils
import DTPM_policies


class PE:
    '''
    A processing element (PE) is the basic resource that defines
    the simpy processes.

    A PE has a *name*, *utilization (float)* and a process (resource)
    '''

    def __init__(self, env, type, name, ID, cluster_ID, capacity, cost):
        '''
        env: Pointer to the current simulation environment
        name: Name of the current processing element
        ID: ID of the current processing element
        capacity: Number tasks that a resource can run simultaneously
        '''
        self.env = env
        self.type = type
        self.name = name
        self.ID = ID
        self.capacity = capacity  # Current capacity of the PE (depends on the number of active cores)
        self.total_capacity = capacity  # Total capacity of the PE
        self.cluster_ID = cluster_ID

        self.enabled = True  # Indicate if the PE is ON
        self.utilization = 0  # Describes how much one PE is utilized
        self.utilization_list = []  # List containing the PE utilization for each sample inside a snippet
        self.current_power_active_core = 0  # Indicate the current power for the active cores (dynamic + static)
        self.current_leakage_core = 0  # Indicate the current leakage power
        self.snippet_energy = 0  # Indicate the energy consumption of the current snippet
        self.total_energy = 0  # Indicate the total energy consumed by the given PE

        self.Cdyn_alpha = 0  # Variable that stores the dynamic capacitance * switching activity for each PE

        self.queue = []  # List of currently running task on a PE
        self.available_time = 0  # Estimated available time of the PE
        self.available_time_list = [0] * self.capacity  # Estimated available time for each core os the PE
        self.idle = True  # The variable indicates whether the PE is active or not

        self.info = []  # List to record all the events happened on a PE
        self.process = simpy.Resource(env, capacity=self.capacity)
        self.cost = cost

        if (common.DEBUG_CONFIG):
            print('[D] Constructed PE-%d with name %s' % (ID, name))

    # Start the "run" process for this PE
    # 在DASH_Sim_core.run中调用
    # self, executable_task, current_resource, DTPM_module
    def run(self, sim_manager, task, resource, DVFS_module):
        '''
        Run this PE to execute a given task.
        The execution time is retrieved from resource_matrix and task name
        '''
        try:
            with self.process.request() as req:  # Requesting the resource for the task
                yield req
                if common.ClusterManager.cluster_list[
                    self.cluster_ID].current_frequency == 0:  # Initialize the frequency if it was not set yet
                    # Depending on the DVFS policy on this PE, set the initial frequency and voltage accordingly
                    if common.ClusterManager.cluster_list[self.cluster_ID].DVFS != 'none' or len(
                            common.ClusterManager.cluster_list[self.cluster_ID].OPP) != 0:
                        DTPM_policies.initialize_frequency(common.ClusterManager.cluster_list[self.cluster_ID])

                        DASH_Sim_utils.trace_frequency(self.env.now,
                                                       common.ClusterManager.cluster_list[self.cluster_ID])

                self.idle = False  # Since the PE starts execution of a task, it is not idle anymore
                common.TaskQueues.running.list.append(
                    task)  # Since the execution started for the task we should add it to the running queue
                task.start_time = self.env.now  # When a resource starts executing the task, record it as the start time
                if common.scheduler == 'SDP_EC':
                    if task.ID in common.sorted_nodes:
                        common.sorted_nodes.remove(task.ID)
                # if this is the leading task of this job,
                # increment the injection counter
                # 如果该task为job中起始任务且当前时间已过冷启动时间
                if (task.head == True) and (self.env.now >= common.warmup_period):
                    # 注入job数+=1
                    common.results.injected_jobs += 1
                    if (common.DEBUG_JOB):
                        print('[D] Time %d: Total injected jobs becomes: %d'
                              % (self.env.now, common.results.injected_jobs))

                    # Store the injected job for validation
                    if (common.simulation_mode == 'validation'):
                        common.Validation.injected_jobs.append(task.jobID)
                # end of if ( (next_task.head == True) and ...

                # 不进入
                if (common.DEBUG_JOB):
                    print('[D] Time %d: Task %s execution is started with frequency %d by PE-%d %s'
                          % (
                              self.env.now, task.ID,
                              common.ClusterManager.cluster_list[self.cluster_ID].current_frequency,
                              self.ID, self.name))

                # Retrieve the execution time and power consumption from the model
                task_runtime_max_freq, randomization_factor = DTPM_power_models.get_execution_time_max_frequency(task,
                                                                                                                 resource)  # Get the run time and power consumption

                dynamic_energy = 0
                static_energy = 0
                task_complete = False
                task_elapsed_time = task.task_elapsed_time_max_freq

                while task_complete is False:
                    # The predicted time takes into account the current frequency and subtracts the time that the task already executed
                    # task预计执行时间
                    predicted_exec_time = (task_runtime_max_freq - task_elapsed_time) + (
                            task_runtime_max_freq - task_elapsed_time) * DTPM_power_models.compute_DVFS_performance_slowdown(
                        common.ClusterManager.cluster_list[self.cluster_ID])
                    # 窗口保留时间
                    window_remaining_time = common.sampling_rate - self.env.now % common.sampling_rate
                    # Test if the task finished before the next sampling period
                    if predicted_exec_time - window_remaining_time > 0:
                        # Run until the next sampling timestamp
                        simulation_step = window_remaining_time
                        slowdown = DTPM_power_models.compute_DVFS_performance_slowdown(
                            common.ClusterManager.cluster_list[self.cluster_ID]) + 1
                        task_elapsed_time += simulation_step / slowdown
                    else:
                        # Run until the task ends
                        simulation_step = predicted_exec_time
                        task_complete = True
                    # if common.scheduler == 'ULS':
                    #     simulation_step = common.table[task.ID][3] - task.start_time
                    # if common.table[task.ID][3] - task.start_time < 0:
                    #     print(task.ID, common.table[task.ID], task.start_time)
                    # Compute the static energy
                    current_leakage = DTPM_power_models.compute_static_power_dissipation(self.cluster_ID)
                    # print(self.name, current_leakage)
                    static_energy += current_leakage * simulation_step * 1e-6

                    max_power_consumption, freq_threshold = DTPM_power_models.get_max_power_consumption(
                        common.ClusterManager.cluster_list[self.cluster_ID],
                        sim_manager.PEs)  # of this task on this resource running at max frequency
                    # Based on the total power consumption and the leakage, get the dynamic power
                    if max_power_consumption > 0:
                        dynamic_power_cluster = max_power_consumption - current_leakage * len(
                            common.ClusterManager.cluster_list[self.cluster_ID].power_profile[freq_threshold])
                        # After obtaining the dynamic power for the cluster, divide it by the number of cores being used to get the power per core
                        dynamic_power_max_freq_core = dynamic_power_cluster / DASH_Sim_utils.get_num_tasks_being_executed(
                            common.ClusterManager.cluster_list[self.cluster_ID], sim_manager.PEs)
                    else:
                        dynamic_power_max_freq_core = 0

                    # Compute the capacitance and alpha based on the dynamic power
                    self.Cdyn_alpha = DTPM_power_models.compute_Cdyn_and_alpha(resource, dynamic_power_max_freq_core,
                                                                               freq_threshold)
                    # print(self.cluster_ID, common.ClusterManager.cluster_list[self.cluster_ID].current_frequency, common.ClusterManager.cluster_list[self.cluster_ID].current_voltage, self.Cdyn_alpha)
                    # Compute the dynamic energy
                    dynamic_power = DTPM_power_models.compute_dynamic_power_dissipation(
                        common.ClusterManager.cluster_list[self.cluster_ID].current_frequency,
                        common.ClusterManager.cluster_list[self.cluster_ID].current_voltage,
                        self.Cdyn_alpha)
                    # print(task.ID, common.ClusterManager.cluster_list[self.cluster_ID].current_frequency, common.ClusterManager.cluster_list[self.cluster_ID].current_voltage, self.Cdyn_alpha)
                    # print(task.ID, self.name, dynamic_power, current_leakage, dynamic_power + current_leakage)
                    # print(task.ID, dynamic_power)
                    dynamic_energy += dynamic_power * simulation_step * 1e-6
                    # Scale the power based on the number of active cores
                    common.ClusterManager.cluster_list[
                        self.cluster_ID].current_power_cluster = dynamic_power * DASH_Sim_utils.get_num_tasks_being_executed(
                        common.ClusterManager.cluster_list[self.cluster_ID], sim_manager.PEs) + current_leakage * \
                                                                 common.ClusterManager.cluster_list[
                                                                     self.cluster_ID].num_active_cores
                    # if self.cluster_ID == 1:
                    # print("@", self.cluster_ID)
                    # print(common.ClusterManager.cluster_list[self.cluster_ID].PE_list)
                    # print(common.ClusterManager.cluster_list[self.cluster_ID].current_frequency,
                    #     common.ClusterManager.cluster_list[self.cluster_ID].current_voltage,
                    #     self.Cdyn_alpha)
                    # print(dynamic_power, DASH_Sim_utils.get_num_tasks_being_executed(
                    #     common.ClusterManager.cluster_list[self.cluster_ID], sim_manager.PEs), current_leakage, common.ClusterManager.cluster_list[
                    #                                                  self.cluster_ID].num_active_cores)
                    # print("!!", task.ID)
                    # print(self.cluster_ID, common.ClusterManager.cluster_list[
                    #     self.cluster_ID].current_power_cluster / len(common.ClusterManager.cluster_list[
                    #     self.cluster_ID].PE_list))
                    self.current_leakage_core = current_leakage
                    self.current_power_active_core = dynamic_power + current_leakage

                    # 进入该分支
                    if (common.simulation_mode == "performance" and self.env.now >= common.warmup_period) or common.simulation_mode == "validation":
                        energy_sample = (dynamic_power + current_leakage) * simulation_step * 1e-6
                        # if common.p_p[task.ID] == 0:
                        # common.p_p[task.ID] += dynamic_power + current_leakage
                        # common.p_num[task.ID] += 1
                        self.snippet_energy += energy_sample
                        self.total_energy += energy_sample
                        # print(common.results.cumulative_energy_consumption, energy_sample, common.results.cumulative_energy_consumption + energy_sample)
                        common.results.cumulative_energy_consumption += energy_sample
                    # print(task.ID, simulation_step)
                    # print(task.ID, task.start_time, task.start_time + simulation_step, simulation_step, common.table[task.ID], resource.ID)
                    common.results.cumulative_exe_time_1 += simulation_step
                    yield self.env.timeout(simulation_step)
                    task.task_elapsed_time_max_freq = task_elapsed_time
                    # At each sample:
                    if self.env.now % common.sampling_rate == 0:
                        # Case 1: If the task is not complete, evaluate this PE at this moment
                        if task_complete is False:
                            DVFS_module.evaluate_PE(resource, self, self.env.now)
                # 任务执行结束
                task.finish_time = int(self.env.now)
                # print(task.ID, resource.name)
                if common.scheduler == 'ULS' or common.scheduler == 'SDP_EC':
                    # common.computation.pop(task.ID)
                    del common.computation[task.ID]
                    del common.jobID[task.ID]
                    del common.baseID[task.ID]
                    del common.prednode[task.ID]

                # if common.table[task.ID][3] - task.finish_time > 1 and task.tail:
                #     print(task.ID, task.start_time, task.finish_time, common.table[task.ID], resource.ID)
                # print(task.ID, task.start_time, task.finish_time, common.table[task.ID], resource.ID)

                # if task.ID == 7616 or task.ID == 7617:
                #     print(task.ID,":",task.start_time,",",task.finish_time,',',resource.ID)

                # As the task finished its execution, reset the task time
                task.task_elapsed_time_max_freq = 0

                if common.DEBUG_JOB:
                    print('[D] Time %d: Task %s execution is finished by PE-%d %s'
                          % (self.env.now, task.ID, self.ID, self.name))

                task_time = task.finish_time - task.start_time
                self.idle = True

                # If there are no OPPs in the model, use the measured power consumption from the model
                if len(common.ClusterManager.cluster_list[self.cluster_ID].OPP) == 0:
                    total_energy_task = dynamic_power_max_freq_core * task_time * 1e-6
                else:
                    total_energy_task = dynamic_energy + static_energy
                    # if task_time != 0:
                    #     print(task.ID, resource.name, total_energy_task, task_time, total_energy_task / task_time * 1e6)
                    # else:
                    #     pass
                # print(task.base_ID, task.jobID)
                # print(task.ID, common.table[task.ID][4], common.table[task.ID][3], common.table[task.ID][0])
                # print(task.ID, task.start_time, task.finish_time, resource.ID)
                # print()
                # 如果该task为job终止任务
                if task.tail:
                    # for i in common.p_p.keys():
                    #     common.p_p[i] /= common.p_num[i]
                    # print(common.p_p, common.p_num)
                    if task.jobname not in common.job_statistics:
                        common.job_statistics[task.jobname] = 1
                    else:
                        common.job_statistics[task.jobname] += 1
                    common.num_of_jobs += 1
                    common.num_of_jobs_1 += 1

                    # common.computation_dict.pop(task.jobID)
                    if common.scheduler != 'OBO':
                        if common.deadline_dict[task.jobID] < task.finish_time:
                            common.num_of_out += 1
                            common.num_of_out_1 += 1
                            if task.jobname not in common.overtime_job_statistics:
                                common.overtime_job_statistics[task.jobname] = 1
                                common.overtime_job_time_statistics[task.jobname] = task.finish_time - common.deadline_dict[
                                    task.jobID]
                            else:
                                common.overtime_job_statistics[task.jobname] += 1
                                common.overtime_job_time_statistics[task.jobname] += (
                                        task.finish_time - common.deadline_dict[task.jobID])
                            print('-----------------------')
                            print('ID', task.jobID, ' is out of deadline')
                            print('ID', task.jobID, '(', task.jobname, ') Arrive Time:', common.arrive_time[task.jobID],
                                  ' Actual Finish Time: ', task.finish_time, '   Deadline:',
                                  common.deadline_dict[task.jobID])
                            print('-----------------------')
                        else:
                            print('ID', task.jobID, '(', task.jobname, ') Arrive Time:', common.arrive_time[task.jobID],
                                  ' Actual Finish Time: ', task.finish_time, '   Deadline:',
                                  common.deadline_dict[task.jobID])
                    # print()
                    # 当前job数-=1
                    if common.scheduler == 'ULS' or common.scheduler == 'dynProb':
                        common.computation.pop(task.ID + 1)
                        common.computation.pop(task.ID + 2)
                        for i in common.tasks_dict[task.jobID]:
                            common.table.pop(i)
                        for i in common.TaskQueues.completed.list[:]:
                            if i.ID in common.tasks_dict[task.jobID]:
                                common.TaskQueues.completed.list.remove(i)
                        # print(len(common.TaskQueues.completed.list))
                        del common.tasks_dict[task.jobID]
                        common.arrive_time.pop(task.jobID)
                        common.communication_dict.pop(task.jobID)
                        common.deadline_dict.pop(task.jobID)
                    if common.scheduler == 'SDP_EC':
                        common.computation.pop(task.ID + 1)
                        common.computation.pop(task.ID + 2)
                        # for i in common.tasks_dict[task.jobID]:
                        #     if i in common.table:
                        #         common.table.pop(i)
                        for i in common.TaskQueues.completed.list[:]:
                            if i.ID in common.tasks_dict[task.jobID]:
                                common.TaskQueues.completed.list.remove(i)
                        # print(len(common.TaskQueues.completed.list))
                        del common.tasks_dict[task.jobID]
                        common.arrive_time.pop(task.jobID)
                        # common.texit.pop(task.jobID)
                        common.communication_dict.pop(task.jobID)
                        common.deadline_dict.pop(task.jobID)
                        common.power.pop(task.jobID)
                    if common.scheduler == 'HEFT_RT':
                        common.arrive_time.pop(task.jobID)
                        common.deadline_dict.pop(task.jobID)
                    common.results.job_counter -= 1
                    common.num_of_jobs_same_time -= 1

                    # 更新completed_queue
                    if common.simulation_mode == 'performance':
                        sim_manager.update_completed_queue()

                    if self.env.now >= common.warmup_period:
                        common.results.execution_time = self.env.now
                        common.results.completed_jobs += 1

                        # Interrupts the timeout of job generator if the inject_jobs_ASAP flag is active
                        # 不进入该分支
                        if sim_manager.job_gen.generate_job and common.inject_jobs_ASAP:
                            sim_manager.job_gen.action.interrupt()

                        # 遍历completed.list
                        for completed in common.TaskQueues.completed.list:
                            # 找到与当前终止节点同一个job的起始节点
                            if (completed.head == True) and (completed.jobID == task.jobID):
                                # 累计执行时间
                                common.results.cumulative_exe_time += (self.env.now - completed.job_start)

                                if (common.DEBUG_JOB):
                                    print('[D] Time %d: Job %d is completed' % (self.env.now, task.jobID + 1))
                    # print('[D] total completed jobs becomes: %d' %(common.results.completed_jobs))
                    # print('[D] Cumulative execution time: %f' %(common.results.cumulative_exe_time))

                    # Store the completed job for validation
                    if (common.simulation_mode == 'validation'):
                        common.Validation.completed_jobs.append(task.jobID)
                # end of if ((task.tail) and ...

                if (common.INFO_SIM):
                    print('[I] Time %d: Task %s is finished by PE-%d %s with %.2f us and energy consumption %.2f J'
                          % (
                              self.env.now, task.ID, self.ID, self.name, round(task_time, 2),
                              round(total_energy_task, 2)))
                # 无用
                DASH_Sim_utils.trace_tasks(task, self, task_time, total_energy_task)
                # for i, executable_task in enumerate(common.TaskQueues.executable.list):
                #    print('Task %d can be executed on PE-%d after time %d'%(executable_task.ID, executable_task.PE_ID, executable_task.time_stamp))

                # Retrieve the energy consumption for the task
                # that the PE just finished processing
                # 累计能量消耗
                common.results.energy_consumption += total_energy_task

                # Since the current task is processed, it should be removed
                # from the outstanding task queue
                # 在completed.list中添加该task，该task完成后根据新的依赖关系更新ready.list
                sim_manager.update_ready_queue(task)

                # Case 2: Evaluate the PE after the queues are updated
                if self.env.now % common.sampling_rate == 0:
                    DVFS_module.evaluate_PE(resource, self, self.env.now)

                # 如果该task为终止节点，且当前时间超过冷启动时间，且当前完成的job数目为10的整数倍
                if task.tail and (self.env.now >= common.warmup_period) and (
                        common.results.completed_jobs % common.snippet_size == 0):
                    # Save new snippet to the dataset
                    DASH_Sim_utils.create_dataset_IL_DTPM(self.env.now, sim_manager.PEs)

                    if common.enable_real_time_constraints:
                        snippet_exec_time = (self.env.now - common.snippet_start_time) * 1e-6
                        snippet_deadline = common.deadline_dict[str(common.current_job_list)]
                        if snippet_exec_time > snippet_deadline:
                            common.missed_deadlines += 1

                    # Reset energy of the snippet
                    for PE in sim_manager.PEs:
                        PE.snippet_energy = 0

                    common.snippet_start_time = self.env.now
                    common.snippet_initial_temp = copy.deepcopy(common.current_temperature_vector)

                    common.snippet_throttle = -1
                    for cluster in common.ClusterManager.cluster_list:
                        cluster.snippet_power_list = []
                    common.snippet_temp_list = []
                    common.snippet_ID_exec += 1
                    if common.job_list != []:
                        if common.snippet_ID_exec < common.max_num_jobs / common.snippet_size:
                            common.current_job_list = common.job_list[common.snippet_ID_exec]

                    # Ends the simulation if all jobs are executed (if sim_early_stop is enabled)
                    if common.results.completed_jobs == common.max_num_jobs:
                        common.time_at_sim_termination = self.env.now
                        sim_manager.sim_done.succeed()
                if task.tail:
                    if common.num_of_jobs_1 == common.simulation_num:
                        print('[I] Number of injected jobs: %d' %(common.results.injected_jobs))
                        print('[I] Number of completed jobs: %d' %(common.results.completed_jobs))
                        print('[I] Number of jobs: %d' %(common.num_of_jobs_1))
                        print('[I] Number of succ jobs: %d' %(common.num_of_jobs_1 - common.num_of_out_1))
                        try:
                            print('[I] Ave latency: %f'
                            %(common.results.cumulative_exe_time/common.results.completed_jobs))
                        except ZeroDivisionError:
                            print('[I] No completed jobs')
                        print("[I] %-30s : %-20s" % ("Execution time(us)", round(common.results.execution_time - common.warmup_period, 2)))
                        print("[I] %-30s : %-20s" % ("Cumulative Execution time(us)", round(common.results.cumulative_exe_time, 2)))
                        print("[I] %-30s : %-20s" % ("Total energy consumption(J)",
                                                     round(common.results.cumulative_energy_consumption, 6)))
                        print("[I] %-30s : %-20s" % ("EDP",
                                                     round((common.results.execution_time - common.warmup_period) * common.results.cumulative_energy_consumption, 2)))
                        print("[I] %-30s : %-20s" % ("Cumulative EDP",
                                                     round(common.results.cumulative_exe_time * common.results.cumulative_energy_consumption, 2)))
                        # print("[I] %-30s : %-20s" % ("Cumulative EDDP",
                        #                              round(common.results.cumulative_exe_time * common.results.cumulative_exe_time * common.results.cumulative_energy_consumption, 2)))
                        print("[I] %-30s : %-20s" % ("Average concurrent jobs", round(common.results.average_job_number, 2)))
                        print("[I] %-30s : %-20s" % ("exe_time:", float(common.exe_time / common.simulation_num)))
                        print("[I] %-30s : %-20s" % ("task_num:", common.task_num))
                        print("[I] %-30s : %-20s" % ("exe_time_node:", float(common.exe_time / common.task_num)))
                        if common.scheduler != 'OBO':
                            sum = 0
                            for i in common.overtime_job_time_statistics.items():
                                sum += i[1]
                            print("[I] %-30s : %-20s" % ("Total Delay:", sum))
                            # self.output('ot', round(sum,3))
                        else:
                            print("[I] %-30s : %-20s" % ("Total Delay:", round(common.overtime_sum, 3)))
                            # self.output('ot', common.overtime_sum)
                        print("[I] %-30s : %-20s" % ("Execution Time:", float(common.exe_time/1000)))
                        print('Overtime Job Statistics:')
                        print(sorted(common.overtime_job_statistics.items(), key=lambda x: x[0]))
                        print('Overtime Statistics:')
                        print(sorted(common.overtime_job_time_statistics.items(), key=lambda x: x[0]))
                        print(common.exe_time)
                        print(common.times)
                        self.output('runtime', float(common.exe_time / common.simulation_num))
                        # self.output('succ', common.num_of_jobs_1 - common.num_of_out_1)
                        # self.output('acj', round(common.results.average_job_number, 2))
                        # self.output('ec', round(common.results.cumulative_energy_consumption, 4))
                        common.exe_time_tmp.append(float(common.exe_time / common.simulation_num))
                        sys.exit()
                        # self.env.timeout(1000000)
                # end of with self.process.request() as req:

        except simpy.Interrupt:
            print('Expect an interrupt at %s' % (self.env.now))

    # end of def run(self, sim_manager, task, resource):
    def output(self, file_path, value):
        # 定义文件路径
        dic_path = 'C:/Users/32628/Desktop/res/' + common.resource_file + '/' + common.deadline_type + '/' + common.scheduler

        # 初始化文件，如果不存在则创建
        if not os.path.exists(dic_path):
            os.makedirs(dic_path)

        file_path = dic_path  + '/' + file_path + ".txt"
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass

        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()
            indexes = list(map(float, lines[0].strip().split(','))) if lines and lines[0].strip() else []
            values = list(map(float, lines[1].strip().split(','))) if len(lines) > 1 and lines[1].strip() else []

        # 插入新的 index 和 value 到合适的位置
        insert_pos = len(indexes)
        for i, idx in enumerate(indexes):
            if common.lam < idx:
                insert_pos = i
                break

        indexes.insert(insert_pos, common.lam)
        values.insert(insert_pos, value)

        # 截断列表，保持每行最多10个元素
        if len(indexes) > 10:
            indexes = indexes[:10]
            values = values[:10]

        # 更新文件
        with open(file_path, 'w') as file:
            file.write(','.join(map(str, indexes)) + '\n')
            file.write(','.join(map(str, values)) + '\n')

# end class PE(object):
