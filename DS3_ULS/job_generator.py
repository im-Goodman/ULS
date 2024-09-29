'''
Description: This file contains the code for the job generator
'''
import random
import copy
import time
from types import SimpleNamespace

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import simpy
import os
import csv

import DTPM
import DTPM_policies
import DTPM_power_models
from ULS import ULS
from Algorithm_RT.dag_merge import merge_2_dags
from heft import dag_merge as d_h
from heft import heft, gantt, dag_merge
from peft import peft
import common
import DASH_Sim_utils
import CP_models


class JobGenerator:
    '''
    Define the JobGenerator class to handle dynamic job generation
    '''

    def __init__(self, env, resource_matrix, jobs, scheduler, PE_list):
        '''
        env: Pointer to the current simulation environment
        resource_matrix: The data structure that defines power/performance
            characteristics of the PEs for each supported task
        jobs: The list of all jobs given to DASH-Sim
        scheduler: Pointer to the DASH_scheduler
        PE_list: The PEs available in the current SoCs
        '''
        self.env = env
        self.resource_matrix = resource_matrix
        self.jobs = jobs
        self.scheduler = scheduler
        self.PEs = PE_list

        # Initially none of the tasks are outstanding
        common.TaskQueues.outstanding = common.TaskManager()  # List of *all* tasks waiting to be processed

        # Initially none of the tasks are completed
        common.TaskQueues.completed = common.TaskManager()  # List of completed tasks

        # Initially none of the tasks are running on the PEs
        common.TaskQueues.running = common.TaskManager()  # List of currently running tasks

        # Initially none of the tasks are completed
        common.TaskQueues.ready = common.TaskManager()  # List of tasks that are ready for processing

        # Initially none of the tasks are in wait ready queue
        common.TaskQueues.wait_ready = common.TaskManager()  # List of tasks that are waiting for being ready for processing

        # Initially none of the tasks are executable
        common.TaskQueues.executable = common.TaskManager()  # List of tasks that are ready for execution

        self.generate_job = True  # Initially $generate_job is True so that as soon as run function is called
        #   it will start generating jobs
        self.max_num_jobs = common.max_num_jobs  # Number of jobs to be created
        self.generated_job_list = []  # List of all jobs that are generated
        self.offset = 0  # This value will be used to assign correct ID numbers for incoming tasks

        self.action = env.process(self.run())  # Starts the run() method as a SimPy process

    def run(self):
        if self.scheduler.name == 'ULS':
            i = 0  # Initialize the iteration variable
            num_jobs = 0
            count = 0
            summation = 0
            # 确保每次iteration中生成的随机数相同
            np.random.seed(common.iteration)

            if len(DASH_Sim_utils.get_current_job_list()) != len(
                    self.jobs.list) and DASH_Sim_utils.get_current_job_list() != []:
                print(
                    '[E] Time %s: Job_list and task_file configs have different lengths, please check DASH.SoC.**.txt file' % self.env.now)
                sys.exit()
            while self.generate_job:  # Continue generating jobs till #generate_job is False
                if common.results.job_counter >= common.max_jobs_in_parallel or (
                        common.job_list != [] and common.snippet_ID_inj == common.snippet_ID_exec):
                    # yield self.env.timeout(self.wait_time)                          # new job addition will be after this wait time
                    try:
                        yield self.env.timeout(common.simulation_clk)
                    except simpy.exceptions.Interrupt:
                        pass
                else:
                    valid_jobs = []
                    common.current_job_list = DASH_Sim_utils.get_current_job_list()
                    # print(common.current_job_list)
                    for index, job_counter in enumerate(common.job_counter_list):
                        if job_counter < common.current_job_list[index]:
                            valid_jobs.append(index)

                    if valid_jobs != []:
                        selection = np.random.choice(valid_jobs)
                    # 进入该分支
                    else:
                        # selection = 0
                        # selection = np.random.choice([0, 1, 2, 3, 4, 5], 1, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
                        # selection = np.random.choice([0, 1], 1, p=[0.8, 0.2]) #表示按0.8和0.2的概率在0、1中选择一个数字
                        # selection = np.random.choice([0, 1], 1, p=[0.8, 0.2]) # WiFi TX, RX
                        # selection = np.random.choice([0, 1], 1, p=[0.5, 0.5]) # Range Detection, Temporal Mitigation
                        # selection = np.random.choice([0, 1, 2, 3], 1, p=[1 / 4, 1 / 4, 1 / 4, 1 / 4])  # All Apps
                        selection = np.random.choice([0, 1, 2, 3, 4], 1,
                                                     p=common.p)  # All Apps
                        # selection = np.random.choice([0, 1, 2, 3, 4], 1,
                        #                              p=[1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5])  # All Apps
                        # selection = np.random.choice([0, 1, 2, 3, 4, 5], 1, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) # All Apps
                        # selection = np.random.choice([0, 1], 1, p=[0.999, 0.001]) # HEFT Journal validation, TempMit + Radar Correlator
                    # if len(self.generated_job_list) == 1:      # for test
                    #     selection += 1
                    common.a += 1
                    if common.a > common.simulation_num:
                        return
                    common.job_name_temp = self.jobs.list[int(selection)].name
                    self.generated_job_list.append(copy.deepcopy(self.jobs.list[
                                                                     int(selection)]))  # Create each job as a deep copy of the job chosen from job list
                    common.arrive_time[i] = self.env.now
                    common.results.job_counter += 1
                    common.flag_1 = True
                    summation += common.results.job_counter
                    count += 1
                    common.results.average_job_number = summation / count
                    common.num_of_jobs_same_time += 1
                    common.communication_dict[i] = self.generated_job_list[i].comm_vol
                    if common.deadline_type == 'tight':
                        common.s = np.random.choice([0, 1], 1, p=[1 / 5, 4 / 5])
                    elif common.deadline_type == 'loose':
                        common.s = np.random.choice([0, 1], 1, p=[4 / 5, 1 / 5])
                    elif common.deadline_type == 'mix':
                        common.s = np.random.choice([0, 1], 1, p=[1 / 2, 1 / 2])
                    # 新任务的dag图
                    job_dag = nx.DiGraph(self.generated_job_list[i].comm_vol)
                    job_dag.remove_edges_from(
                        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist"
                        [edge for edge in job_dag.edges() if job_dag.get_edge_data(*edge)['weight'] == '0.0']
                    )
                    # 根据之前所有job的dag图节点号数目作为偏移量
                    nx.relabel_nodes(job_dag, lambda idx: idx + self.offset, copy=False)
                    tmp = {i: job_dag.nodes()}
                    common.tasks_dict.update(tmp)
                    computation_dict = {}
                    power_dict = {}
                    common.task_num += len(job_dag.nodes())
                    # Build the updated computation and power matrices that the scheduler will use to determine estimated execution times and power consumption
                    # 循环当前job的dag图中所有节点
                    # 合并新工作流dag与原本的dag
                    merged_dag = merge_2_dags(job_dag, common.curr_dag)
                    for node in merged_dag:
                        curr_task = -1
                        for task_1 in common.TaskQueues.outstanding.list:
                            if task_1.ID == node:
                                curr_task = task_1
                        for task_1 in common.TaskQueues.ready.list:
                            if task_1.ID == node:
                                curr_task = task_1
                        for task_1 in common.TaskQueues.executable.list:
                            if task_1.ID == node:
                                curr_task = task_1
                        power_dict[node] = []
                        # 循环cluster
                        for cluster in common.ClusterManager.cluster_list:
                            # 当前cluster所用power
                            cluster_power = cluster.current_power_cluster
                            # 循环当前cluster中processor，此时为processor_id
                            for resource_idx in cluster.PE_list:
                                # if flag == True:
                                #     print(resource_idx)
                                # 获得该processor
                                resource = self.resource_matrix.list[resource_idx]
                                # 获得该节点在job中对应的真实task
                                if curr_task == -1:
                                    # print(node)
                                    associated_task = [task for task in self.generated_job_list[i].task_list if
                                                       task.ID == node - self.offset]
                                    # print(associated_task[0].ID)
                                else:
                                    associated_task = [curr_task]
                                # 如果找得到对应task,且当前processor可以处理该task
                                if len(associated_task) > 0 and associated_task[
                                    0].name in resource.supported_functionalities:
                                    # 获得resource_matrix中supported_functionalities数组对应该node名字的index
                                    perf_index = resource.supported_functionalities.index(associated_task[0].name)
                                    # 计算得到power_dict对应数据
                                    power_dict[node].append(cluster_power / len(cluster.PE_list))
                                # 否则,记无穷大(基本是associated_task[0].name not in resource.supported_functionalities的情况)
                                else:
                                    power_dict[node].append(np.inf)

                    for node in job_dag:
                        computation_dict[node] = []
                        # power_dict[node] = []
                        # 循环cluster
                        for cluster in common.ClusterManager.cluster_list:
                            # 当前cluster所用power
                            cluster_power = cluster.current_power_cluster
                            # 循环当前cluster中processor，此时为processor_id
                            for resource_idx in cluster.PE_list:
                                # if flag == True:
                                #     print(resource_idx)
                                # 获得该processor
                                resource = self.resource_matrix.list[resource_idx]
                                # 获得该节点在job中对应的真实task
                                associated_task = [task for task in self.generated_job_list[i].task_list if
                                                   task.ID == node - self.offset]
                                # 如果找得到对应task,且当前processor可以处理该task
                                if len(associated_task) > 0 and associated_task[
                                    0].name in resource.supported_functionalities:
                                    # 获得resource_matrix中supported_functionalities数组对应该node名字的index
                                    perf_index = resource.supported_functionalities.index(associated_task[0].name)
                                    # 根据index得到对应performance数组中表示该processor执行该task所需的时间
                                    computation_dict[node].append(resource.performance[perf_index])
                                    # 计算得到power_dict对应数据
                                    # power_dict[node].append(cluster_power / len(cluster.PE_list))
                                # 否则,记无穷大(基本是associated_task[0].name not in resource.supported_functionalities的情况)
                                else:
                                    computation_dict[node].append(np.inf)
                                    # power_dict[node].append(np.inf)
                    computation_dict[max(merged_dag) - 1] = np.zeros((1, len(self.scheduler.resource_matrix.list)))
                    computation_dict[max(merged_dag)] = np.zeros((1, len(self.scheduler.resource_matrix.list)))
                    common.computation.update(computation_dict)
                    # 给新增的起始节点和终止节点的power_dict添加对应数量的0
                    power_dict[max(merged_dag) - 1] = [0] * len(self.scheduler.resource_matrix.list)
                    power_dict[max(merged_dag)] = [0] * len(self.scheduler.resource_matrix.list)
                    common.power.update(power_dict)

                    computation_matrix = np.empty((max(common.computation) + 1,
                                                   len(self.resource_matrix.list)))  # Number of nodes * number of resources
                    for key, val in common.computation.items():
                        computation_matrix[key, :] = val
                    power_matrix = np.empty((max(common.computation) + 1,
                                             len(self.resource_matrix.list)))  # Number of nodes * number of resources
                    if common.use_adaptive_scheduling:
                        if common.results.job_counter == common.max_jobs_in_parallel:
                            # System is oversubscribed, use EFT scheduling
                            rank_metric = heft.RankMetric.MEAN
                            op_mode = heft.OpMode.EFT
                        else:
                            # System isn't oversubscribed, use EDP scheduling
                            rank_metric = heft.RankMetric.EDP
                            op_mode = heft.OpMode.EDP_REL
                    # 此时进这个分支
                    else:
                        # RankMetric.MEAN
                        rank_metric = heft.RankMetric(
                            common.config.get('SCHEDULER PARAMETERS', 'heft_rankMetric', fallback='MEAN'))
                        # OpMode.EFT
                        op_mode = heft.OpMode(common.config.get('SCHEDULER PARAMETERS', 'heft_opMode', fallback='EFT'))
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
                        running_tasks[proc].append(ULS.ScheduleEvent(task_id, task_start, task_end, proc))

                    for ii in range(len(self.generated_job_list[i].task_list)):  # Go over each task in the job
                        # 取task
                        next_task = self.generated_job_list[i].task_list[ii]
                        # jobID为调度过程中的job的index
                        next_task.jobID = i  # assign job id to the next task
                        # base_ID为调度过程每个job中task的index
                        next_task.base_ID = ii  # also record the original ID of the next task
                        # ID为在首个job首个task为基础上的真正该task的index
                        next_task.ID = ii + self.offset  # and change the ID of the task accordingly
                        tmp = {next_task.ID: i}
                        common.jobID.update(tmp)
                        tmp = {next_task.ID: ii}
                        common.baseID.update(tmp)
                        # 若该task为job的起始
                        if next_task.head:
                            # 该task所在job的起始时间=该task的起始时间为当下时刻
                            next_task.job_start = self.env.now  # When a new job is generated, its execution is also started
                            self.generated_job_list[i].head_ID = next_task.ID

                        next_task.head_ID = self.generated_job_list[i].head_ID

                        # 相应增加新task的前驱节点的index
                        for k in range(len(next_task.predecessors)):
                            next_task.predecessors[
                                k] += self.offset  # also change the predecessors of the newly added task, accordingly
                        tmp = {next_task.ID: next_task.predecessors}
                        common.prednode.update(tmp)
                        # 如果该节点有前驱节点，则放入等待队列
                        if len(next_task.predecessors) > 0:
                            common.TaskQueues.outstanding.list.append(
                                next_task)  # Add the task to the outstanding queue since it has predecessors
                            # Next, print debug messages
                            if common.DEBUG_SIM:
                                print('[D] Time %d: Adding task %d to the outstanding queue,'
                                      % (self.env.now, next_task.ID), end='')
                                print(' task %d has predecessors:'
                                      % next_task.ID, next_task.predecessors)

                        # 如果该节点无前驱节点，则放入就绪队列
                        else:
                            common.TaskQueues.ready.list.append(
                                next_task)  # Add the task to the ready queue since it has no predecessors
                            if common.DEBUG_SIM:
                                print('[D] Time %s: Task %s is pushed to the ready queue list'
                                      % (self.env.now, next_task.ID), end='')
                                print(', the ready queue list has %s tasks'
                                      % (len(common.TaskQueues.ready.list)))
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
                    common.running.clear()
                    for mm in common.TaskQueues.running.list:
                        common.running.append(mm.ID)
                    for mm in common.TaskQueues.completed.list:
                        common.running.append(mm.ID)
                    if common.job_name_temp == 'TEST':
                        deadline = 35
                    elif common.job_name_temp == 'TEST1':
                        deadline = 48
                    common.deadline_dict[i] = deadline + self.env.now
                    if running_tasks is None:
                        running_tasks = {}
                    over_time = []
                    for node in merged_dag:
                        if 'sd' in merged_dag.nodes()[node]:
                            task = -1
                            for task_1 in common.TaskQueues.ready.list:
                                if task_1.ID == node:
                                    task = task_1
                            for task_1 in common.TaskQueues.executable.list:
                                if task_1.ID == node:
                                    task = task_1
                            for task_1 in common.TaskQueues.outstanding.list:
                                if task_1.ID == node:
                                    task = task_1
                            if common.deadline_dict[task.jobID] <= self.env.now:
                                over_time.append(node)
                    dict1 = {}
                    for node in merged_dag.nodes:
                        for task_1 in common.TaskQueues.ready.list:
                            if task_1.ID == node:
                                dict1[node] = task_1
                                break
                        for task_1 in common.TaskQueues.executable.list:
                            if task_1.ID == node:
                                dict1[node] = task_1
                                break
                        for task_1 in common.TaskQueues.outstanding.list:
                            if task_1.ID == node:
                                dict1[node] = task_1
                                break
                    dict2 = {}
                    for node in merged_dag.nodes:
                        for task_1 in common.TaskQueues.completed.list:
                            if task_1.ID == node:
                                dict2[node] = task_1
                                break
                    _self = {
                        'computation_matrix': computation_matrix,
                        'communication_matrix': common.ResourceManager.comm_band,
                        'task_schedules': {},
                        'prev': [],
                        'proc_schedules': running_tasks,
                        'numExistingJobs': 0,
                        'time_offset': self.env.now,
                        'PEs': self.PEs,
                        'root_node': None,
                        'terminal_node': None,
                        'deadline': deadline,
                        'jobID': i,
                        'avgComm': 0,
                        'over_time': over_time,
                        'dict1' : dict1,
                        'dict2' : dict2
                    }
                    with open("C:/Users/32628/Desktop/overtime.txt", 'w') as file:
                        file.write(str(over_time) + '\n')
                    with open("C:/Users/32628/Desktop/dict1.txt", 'w') as file:
                        file.write("dict1:\n")
                        for node in dict1.keys():
                            file.write(f'{node}:{dict1.get(node).jobID} {dict1.get(node).base_ID} {dict1.get(node).predecessors} {dict1.get(node).ID}\n')
                    with open("C:/Users/32628/Desktop/dict2.txt", 'w') as file:
                        file.write("dict2:\n")
                        for node in dict2.keys():
                            file.write(f'{node}:{dict2.get(node).jobID} {dict2.get(node).base_ID} {dict2.get(node).predecessors} {dict2.get(node).ID}\n')
                    _self = SimpleNamespace(**_self)
                    diagonal_mask = np.ones(_self.communication_matrix.shape, dtype=bool)
                    np.fill_diagonal(diagonal_mask, 0)
                    avgCommunicationCost = np.mean(_self.communication_matrix[diagonal_mask])
                    _self.avgComm = avgCommunicationCost
                    root_node = [node for node in merged_dag.nodes() if
                                 not any(True for _ in merged_dag.predecessors(node))]
                    root_node = root_node[0]
                    _self.root_node = root_node
                    terminal_node = [node for node in merged_dag.nodes() if
                                     not any(True for _ in merged_dag.successors(node))]
                    terminal_node = terminal_node[0]
                    _self.terminal_node = terminal_node
                    for edge in merged_dag.edges():
                        nx.set_edge_attributes(merged_dag, {
                            edge: float(merged_dag.get_edge_data(*edge)['weight']) / _self.avgComm},
                                               'avgweight')

                    with open('C:/Users/32628/Desktop/dag.txt', 'w') as file:  # 输出节点
                        file.write("Nodes:\n")
                        for node in merged_dag.nodes():
                            file.write(str(node) + ':' + "{t:" + str(merged_dag.nodes[node]["t"] if "t" in merged_dag.nodes[node] else -1) + "} {t1:" + str(merged_dag.nodes[node]["t1"] if "t1" in merged_dag.nodes[node] else -1) + "} {sd:" + str(merged_dag.nodes[node]["sd"] if "sd" in merged_dag.nodes[node] else -1) + "}\n")

                        # 输出边权值
                        file.write("\nEdge Weights:\n")
                        for edge in merged_dag.edges():
                            weight = merged_dag[edge[0]][edge[1]].get('weight')
                            file.write(f'{edge[0]}-{edge[1]}: {weight}\n')

                        file.write("\nAvg Weights:\n")
                        for edge in merged_dag.edges():
                            weight = merged_dag[edge[0]][edge[1]].get('avgweight')
                            file.write(f'{edge[0]}-{edge[1]}: {weight}\n')

                        # 输出父节点集合
                        file.write("\nParent Sets:\n")
                        for node in merged_dag.nodes():
                            parents = list(merged_dag.predecessors(node))
                            parent_str = ','.join(map(str, parents)) if parents else 'null'
                            file.write(f'{node}:{parent_str}\n')

                        file.write("\nSuccessor Sets:\n")
                        for node in merged_dag.nodes():
                            successor = list(merged_dag.successors(node))
                            successor_str = ','.join(map(str, successor)) if successor else 'null'
                            file.write(f'{node}:{successor_str}\n')

                    with open('C:/Users/32628/Desktop/_self.txt', 'w') as file:  # 输出节点
                        file.write("communication:\n")
                        for row in common.ResourceManager.comm_band:
                            # 将每一行转换为字符串，并写入文件
                            file.write(' '.join(map(str, row)) + '\n')

                        file.write("\ncommunication1:\n")
                        for job in common.communication_dict.keys():
                            file.write(str(job) + '\n')
                            for row in common.communication_dict[job]:
                                # 将每一行转换为字符串，并写入文件
                                file.write(' '.join(map(str, row)) + '\n')

                        file.write("\ndeadline_dict:\n")
                        for job in common.deadline_dict.keys():
                            file.write(str(job) + ":" + str(common.deadline_dict[job]) + '\n')

                        # 输出边权值
                        file.write("\nproc_schedules:\n")
                        for proc in running_tasks.keys():
                            file.write(f'{proc}:{running_tasks.get(proc)}\n')

                        file.write("\ntable:\n")
                        for node in common.table.keys():
                            file.write(f'{node}:{common.table.get(node)}\n')

                        # 输出父节点集合
                        file.write("\ncomputation:\n")
                        # 遍历数组的每一行
                        for row in computation_matrix:
                            # 将每一行转换为字符串，并写入文件
                            file.write(' '.join(map(str, row)) + '\n')

                        file.write("\ntexit:\n")
                        for jobId in common.texit.keys():
                            file.write(f'{jobId}:{common.texit.get(jobId)}\n')

                        file.write("\nrunning:\n")
                        # 遍历数组的每一行
                        file.write(str(common.running) + '\n')

                        file.write("\nroot_node:\n")
                        file.write(str(_self.root_node) + "\n")

                        file.write("\nterminal_node:\n")
                        file.write(str(_self.terminal_node) + "\n")

                        file.write("\noffset:\n")
                        file.write(str(_self.time_offset) + "\n")

                        file.write("\njobId:\n")
                        file.write(str(i) + "\n")

                        file.write("\ndeadline:\n")
                        file.write(str(_self.deadline) + "\n")

                        file.write("\navgComm:\n")
                        file.write(str(_self.avgComm) + "\n")


                    proc_sched, task_sched, dict_output = ULS.schedule_dag(
                        merged_dag,
                        _self,
                        table=common.table,
                        communication_dict=common.communication_dict,
                        communication_matrix=common.ResourceManager.comm_band,
                        proc_schedules=running_tasks,
                        relabel_nodes=False,
                        rank_metric=rank_metric,
                        power_dict=common.power
                    )

                    # print(task_sched)

                    if self.scheduler.name != 'SDP_EC':
                        if isinstance(common.table, dict):
                            for key, value in dict_output.items():
                                # if key not in common.table.keys():
                                common.table[key] = value
                        else:
                            common.table = dict_output
                        # print(common.table)
                        common.proc_schedule = proc_sched
                    common.computation_dict = computation_dict
                    common.curr_dag = merged_dag
                    # offset根据新增job的task数增加
                    self.offset += len(self.generated_job_list[i].task_list) + 2
                    common.offset = self.offset

                    # Update the job ID
                    i += 1
                    common.warmup_period = 0
                    # common.warmup_period = 10000
                    # 当前时间大于10000或 'validation'
                    if self.env.now > common.warmup_period or common.simulation_mode == 'validation':
                        num_jobs += 1
                        # 不进入该分支
                        if common.job_counter_list:
                            common.job_counter_list[selection] += 1
                            count_complete_jobs = 0
                            # Check if all jobs for the current snippet were injected
                            common.current_job_list = DASH_Sim_utils.get_current_job_list()
                            for index, job_counter in enumerate(common.job_counter_list):
                                if job_counter == common.current_job_list[index]:
                                    count_complete_jobs += 1
                            if count_complete_jobs == len(common.job_counter_list) and num_jobs < common.max_num_jobs:
                                # Get the next snippet's job list
                                common.snippet_ID_inj += 1
                                np.random.seed(common.iteration)
                                common.job_counter_list = [0] * len(common.current_job_list)

                    # 不进入此分支
                    if common.fixed_injection_rate:
                        self.wait_time = common.scale
                    # 进入此分支
                    else:
                        self.wait_time = int(random.expovariate(
                            1 / common.scale))  # assign an exponentially distributed random variable to $wait_time

                    try:
                        random.seed(common.iteration)
                        # a = random.randint(35, 50)
                        # a = random.randint(40, 60) # EC CRWD
                        # a = random.expovariate(0.035)  # 97 92
                        a = random.expovariate(common.lam)  # EC
                        # a = random.randint(50, 80)  # 99 95
                        # yield self.env.timeout(a)
                        yield self.env.timeout(100)
                        # yield self.env.timeout(self.wait_time)                          # new job addition will be after this wait time
                    except simpy.exceptions.Interrupt:
                        pass

                # end of while (self.generate_job):
        else:
            i = 0  # Initialize the iteration variable
            num_jobs = 0
            count = 0
            summation = 0
            # 确保每次iteration中生成的随机数相同
            np.random.seed(common.iteration)

            if len(DASH_Sim_utils.get_current_job_list()) != len(
                    self.jobs.list) and DASH_Sim_utils.get_current_job_list() != []:
                print(
                    '[E] Time %s: Job_list and task_file configs have different lengths, please check DASH.SoC.**.txt file'
                    % self.env.now)
                sys.exit()
            while self.generate_job:  # Continue generating jobs till #generate_job is False
                if (common.results.job_counter >= common.max_jobs_in_parallel or (
                        common.job_list != [] and common.snippet_ID_inj == common.snippet_ID_exec)):
                    # yield self.env.timeout(self.wait_time)                          # new job addition will be after this wait time
                    try:
                        yield self.env.timeout(common.simulation_clk)
                    except simpy.exceptions.Interrupt:
                        pass
                else:
                    valid_jobs = []
                    common.current_job_list = DASH_Sim_utils.get_current_job_list()
                    for index, job_counter in enumerate(common.job_counter_list):
                        if job_counter < common.current_job_list[index]:
                            valid_jobs.append(index)

                    if valid_jobs:
                        selection = np.random.choice(valid_jobs)
                    # 进入该分支
                    else:
                        selection = 0
                        # selection = np.random.choice([0, 1, 2, 3, 4], 1,
                        #                              p=[0.1, 0.1, 0.5, 0.2, 0.1])
                        # selection = np.random.choice([0, 1, 2, 3, 4, 5], 1, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
                        # selection = np.random.choice([0, 1], 1, p=[0.8, 0.2]) #表示按0.8和0.2的概率在0、1中选择一个数字
                        # selection = np.random.choice([0, 1], 1, p=[0.8, 0.2]) # WiFi TX, RX
                        # selection = np.random.choice([0, 1], 1, p=[0.8, 0.2]) # Range Detection, Temporal Mitigation
                        # selection = np.random.choice([0, 1, 2, 3], 1, p=[1 / 4, 1 / 4, 1 / 4, 1 / 4])  # All Apps
                        selection = np.random.choice([0, 1, 2, 3, 4], 1,
                                                     p=common.p)  # All Apps
                        # p=[0.05, 0.025, 0.6, 0.2, 0.125]
                        # p=[0.05, 0.05, 0.6, 0.25, 0.05]  p=[0.05, 0.05, 0.5, 0.25, 0.15] p=[0.1, 0.05, 0.6, 0.2, 0.05]
                        # selection = np.random.choice([0, 1, 2, 3, 4, 5], 1, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) # All Apps
                        # selection = np.random.choice([0, 1], 1, p=[0.999, 0.001]) # HEFT Journal validation, TempMit + Radar Correlator
                        # print('selected job id is',selection)
                    # if len(self.generated_job_list) == 1:  # for test
                    #     selection += 1
                    common.a += 1
                    if common.a > common.simulation_num:
                        return
                    common.job_name_temp = self.jobs.list[int(selection)].name
                    # print(common.job_name_temp, self.env.now)
                    self.generated_job_list.append(copy.deepcopy(self.jobs.list[
                                                                     int(selection)]))  # Create each job as a deep copy of the job chosen from job list
                    common.task_num += len(self.generated_job_list[i].task_list)
                    common.arrive_time[i] = self.env.now
                    common.results.job_counter += 1
                    summation += common.results.job_counter
                    count += 1
                    if common.deadline_type == 'tight':
                        common.s = np.random.choice([0, 1], 1, p=[1 / 5, 4 / 5])
                    elif common.deadline_type == 'loose':
                        common.s = np.random.choice([0, 1], 1, p=[4 / 5, 1 / 5])
                    elif common.deadline_type == 'mix':
                        common.s = np.random.choice([0, 1], 1, p=[1 / 2, 1 / 2])
                    common.results.average_job_number = summation / count
                    if common.DEBUG_JOB:
                        print('[D] Time %d: Job generator added job %d' % (self.env.now, i + 1))

                    if common.simulation_mode == 'validation':
                        common.Validation.generated_jobs.append(i)

                    # Should this move in a "if (common.scheduler_type == DAG scheduler)" direction?
                    if self.scheduler.name == 'OBO' or self.scheduler.name == 'PEFT':
                        # Load the graph associated with this job
                        # 当前到来的job的dag图
                        job_dag = nx.DiGraph(self.generated_job_list[i].comm_vol)
                        job_dag.remove_edges_from(
                            # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist"
                            [edge for edge in job_dag.edges() if job_dag.get_edge_data(*edge)['weight'] == '0.0']
                        )
                        # 根据之前所有job的dag图节点号数目作为偏移量
                        nx.relabel_nodes(job_dag, lambda idx: idx + self.offset, copy=False)
                        computation_dict = {}
                        power_dict = {}
                        outstanding_dag = nx.DiGraph()
                        # Build the updated computation and power matrices that the scheduler will use to determine estimated execution times and power consumption
                        flag = True
                        # 循环当前job的dag图中所有节点
                        for node in job_dag:
                            # if flag==True:
                            #     print(node)
                            computation_dict[node] = []
                            power_dict[node] = []
                            # 循环cluster
                            for cluster in common.ClusterManager.cluster_list:
                                # 当前cluster所用power
                                cluster_power = cluster.current_power_cluster
                                # 循环当前cluster中processor，此时为processor_id
                                for resource_idx in cluster.PE_list:
                                    # 获得该processor
                                    resource = self.resource_matrix.list[resource_idx]
                                    # 获得该节点在job中对应的真实task
                                    associated_task = [task for task in self.generated_job_list[i].task_list if
                                                       task.ID == node - self.offset]
                                    # 如果找得到对应task,且当前processor可以处理该task
                                    if len(associated_task) > 0 and associated_task[
                                        0].name in resource.supported_functionalities:
                                        # 获得resource_matrix中supported_functionalities数组对应该node名字的index
                                        perf_index = resource.supported_functionalities.index(associated_task[0].name)
                                        # 根据index得到对应performance数组中表示该processor执行该task所需的时间
                                        computation_dict[node].append(resource.performance[perf_index])
                                        # 计算得到power_dict对应数据
                                        power_dict[node].append(cluster_power / len(cluster.PE_list))
                                    # 否则,记无穷大(基本是associated_task[0].name not in resource.supported_functionalities的情况)
                                    else:
                                        computation_dict[node].append(np.inf)
                                        power_dict[node].append(np.inf)
                        # Build the current list of running or already-scheduled tasks so that the scheduler takes them into account
                        running_tasks = {}
                        for idx in range(len(self.PEs)):
                            running_tasks[idx] = []

                        merge_method = dag_merge.MergeMethod[
                            common.config.get('SCHEDULER PARAMETERS', 'heft_mergeMethod', fallback='COMMON_ENTRY_EXIT')]

                        # 此时不走这分支
                        if common.use_adaptive_scheduling:
                            if common.results.job_counter == common.max_jobs_in_parallel:
                                # System is oversubscribed, use EFT scheduling
                                rank_metric = heft.RankMetric.MEAN
                                op_mode = heft.OpMode.EFT
                            else:
                                # System isn't oversubscribed, use EDP scheduling
                                rank_metric = heft.RankMetric.EDP
                                op_mode = heft.OpMode.EDP_REL
                        # 此时进这个分支
                        else:
                            # RankMetric.MEAN
                            rank_metric = heft.RankMetric(
                                common.config.get('SCHEDULER PARAMETERS', 'heft_rankMetric', fallback='MEAN'))
                            # OpMode.EFT
                            op_mode = heft.OpMode(
                                common.config.get('SCHEDULER PARAMETERS', 'heft_opMode', fallback='EFT'))

                        # Remove placeholder source/sink nodes from the last time the graph was merged so they don't continually accumulate with each iteration
                        if len(outstanding_dag) is not 0:
                            outstanding_dag.remove_node(max(outstanding_dag) - 1)
                            outstanding_dag.remove_node(max(outstanding_dag))
                            # 对初始图进行操作，添加新的起始和终止节点
                            merged_dag = heft.dag_merge.merge_dags(outstanding_dag, job_dag,
                                                                   merge_method=heft.dag_merge.MergeMethod.COMMON_ENTRY_EXIT,
                                                                   skip_relabeling=True)
                        # 对初始图进行操作，添加新的起始和终止节点
                        merged_dag = d_h.merge_dags(outstanding_dag, job_dag,
                                                    merge_method=d_h.MergeMethod.COMMON_ENTRY_EXIT,
                                                    skip_relabeling=True)
                        # print(merged_dag.nodes())
                        # 给新增的起始节点和终止节点的computation_dict添加对应数量的0
                        computation_dict[max(merged_dag) - 1] = [0] * len(self.scheduler.resource_matrix.list)
                        computation_dict[max(merged_dag)] = [0] * len(self.scheduler.resource_matrix.list)
                        # 给新增的起始节点和终止节点的power_dict添加对应数量的0
                        power_dict[max(merged_dag) - 1] = np.zeros((1, len(self.scheduler.resource_matrix.list)))
                        power_dict[max(merged_dag)] = np.zeros((1, len(self.scheduler.resource_matrix.list)))

                        # 不进入该分支
                        if common.DEBUG_SCH:
                            plt.clf()
                            # Requires "pydot" package, available through conda install.
                            nx.draw(merged_dag, pos=nx.nx_pydot.graphviz_layout(merged_dag, prog='dot'),
                                    with_labels=True)
                            plt.show()

                        if common.scheduler != 'OBO':
                            # Load the appropriate parameters for passing this current DAG through HEFT
                            computation_matrix = np.empty((max(merged_dag) + 1,
                                                           len(self.scheduler.resource_matrix.list)))  # Number of nodes * number of resources
                            for key, val in computation_dict.items():
                                computation_matrix[key, :] = val
                        else:
                            computation_matrix = np.empty((max(merged_dag) + 1,
                                                           len(self.scheduler.resource_matrix.list)))  # Number of nodes * number of resources
                            for key, val in computation_dict.items():
                                # print(val)
                                computation_matrix[key, :] = val
                        power_matrix = np.empty((max(merged_dag) + 1,
                                                 len(self.scheduler.resource_matrix.list)))  # Number of nodes * number of resources
                        if self.scheduler.name == 'OBO':
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
                            common.deadline_dict[i] = deadline + self.env.now
                            if i != 0:
                                temp = copy.deepcopy(common.running_task)
                                for proc in range(len(common.ResourceManager.comm_band)):
                                    for sched in temp[proc]:
                                        if sched.end < common.arrive_time[i]:
                                            common.running_task[proc].remove(sched)
                            # print(common.running_task)
                            if common.scheduler == 'OBO':

                                proc_sched, task_sched, dict_output = heft.schedule_dag(
                                    common.job_name_temp,
                                    i,
                                    merged_dag,
                                    computation_matrix=computation_matrix,
                                    communication_matrix=common.ResourceManager.comm_band,
                                    proc_schedules=common.running_task,
                                    time_offset=self.env.now,
                                    relabel_nodes=False,
                                    rank_metric=rank_metric,
                                    power_dict=power_matrix,
                                    op_mode=op_mode
                                )
                        else:
                            proc_sched, task_sched, dict_output = peft.schedule_dag(
                                merged_dag,
                                self.PEs,
                                computation_matrix=computation_matrix,
                                communication_matrix=common.ResourceManager.comm_band,
                                proc_schedules=running_tasks,
                                time_offset=self.env.now,
                                relabel_nodes=False
                            )

                        if common.DEBUG_SCH:
                            print('[D] Predicted HEFT finish time is %f' % task_sched[
                                max(key for key, _ in task_sched.items())].end)
                            # gantt.showGanttChart(proc_sched)

                        # common.table = dict_output
                        if isinstance(common.table, dict):
                            for key, value in dict_output.items():
                                common.table[key] = value
                        else:
                            common.table = dict_output
                        common.current_dag = merged_dag
                        common.computation_dict = computation_dict
                    for ii in range(len(self.generated_job_list[i].task_list)):  # Go over each task in the job
                        # 取task
                        next_task = self.generated_job_list[i].task_list[ii]
                        # jobID为调度过程中的job的index
                        next_task.jobID = i  # assign job id to the next task
                        # base_ID为调度过程每个job中task的index
                        next_task.base_ID = ii  # also record the original ID of the next task
                        # ID为在首个job首个task为基础上的真正该task的index
                        next_task.ID = ii + self.offset  # and change the ID of the task accordingly

                        # 若该task为job的起始
                        if next_task.head:
                            # 该task所在job的起始时间=该task的起始时间为当下时刻
                            next_task.job_start = self.env.now  # When a new job is generated, its execution is also started
                            self.generated_job_list[i].head_ID = next_task.ID

                        next_task.head_ID = self.generated_job_list[i].head_ID

                        # 相应增加新task的前驱节点的index
                        for k in range(len(next_task.predecessors)):
                            next_task.predecessors[
                                k] += self.offset  # also change the predecessors of the newly added task, accordingly

                        # 如果该节点有前驱节点，则放入等待队列
                        if len(next_task.predecessors) > 0:
                            common.TaskQueues.outstanding.list.append(
                                next_task)  # Add the task to the outstanding queue since it has predecessors
                            # Next, print debug messages
                            if common.DEBUG_SIM:
                                print('[D] Time %d: Adding task %d to the outstanding queue,'
                                      % (self.env.now, next_task.ID), end='')
                                print(' task %d has predecessors:'
                                      % (next_task.ID), next_task.predecessors)

                        # 如果该节点无前驱节点，则放入就绪队列
                        else:
                            common.TaskQueues.ready.list.append(
                                next_task)  # Add the task to the ready queue since it has no predecessors
                            if common.DEBUG_SIM:
                                print('[D] Time %s: Task %s is pushed to the ready queue list'
                                      % (self.env.now, next_task.ID), end='')
                                print(', the ready queue list has %s tasks'
                                      % (len(common.TaskQueues.ready.list)))
                    # offset根据新增job的task数增加
                    if common.scheduler == 'HEFT':
                        self.offset += len(self.generated_job_list[i].task_list) + 2
                    elif common.scheduler == 'Prob':
                        self.offset += len(self.generated_job_list[i].task_list) + 2
                    else:
                        self.offset += len(self.generated_job_list[i].task_list)
                    # end of for ii in range(len(self.generated_job_list[i].list))

                    # 不进入该分支
                    if 'CP' in self.scheduler.name:
                        while len(common.TaskQueues.executable.list) > 0:
                            task = common.TaskQueues.executable.list.pop(-1)
                            common.TaskQueues.ready.list.append(task)

                        if self.scheduler.name == 'CP_CLUSTER':
                            CP_models.CP_Cluster(self.env.now, self.PEs, self.resource_matrix, self.jobs,
                                                 self.generated_job_list)
                        if self.scheduler.name == 'CP_PE':
                            CP_models.CP_PE(self.env.now, self.PEs, self.resource_matrix, self.jobs,
                                            self.generated_job_list)
                        if self.scheduler.name == 'CP_MULTI':
                            CP_models.CP_Multi(self.env.now, self.PEs, self.resource_matrix, self.jobs,
                                               self.generated_job_list)

                    # Update the job ID
                    i += 1
                    # common.warmup_period = 10000
                    # 当前时间大于10000或 'validation'
                    if self.env.now > common.warmup_period or common.simulation_mode == 'validation':
                        num_jobs += 1
                        # 不进入该分支
                        if common.job_counter_list:
                            common.job_counter_list[selection] += 1
                            count_complete_jobs = 0
                            # Check if all jobs for the current snippet were injected
                            common.current_job_list = DASH_Sim_utils.get_current_job_list()
                            for index, job_counter in enumerate(common.job_counter_list):
                                if job_counter == common.current_job_list[index]:
                                    count_complete_jobs += 1
                            if count_complete_jobs == len(common.job_counter_list) and num_jobs < common.max_num_jobs:
                                # Get the next snippet's job list
                                common.snippet_ID_inj += 1
                                np.random.seed(common.iteration)
                                common.job_counter_list = [0] * len(common.current_job_list)

                    # 不进入此分支
                    if common.simulation_mode == 'validation' or common.sim_early_stop:
                        if (
                                num_jobs >= self.max_num_jobs):  # check if max number of jobs, given in config file, are created
                            self.generate_job = False  # if yes, no more jobs will be added to simulation

                    # print ('lambda value is: %.2f' %(1/common.scale))
                    # 不进入此分支
                    if common.fixed_injection_rate:
                        self.wait_time = common.scale
                    # 进入此分支
                    else:
                        self.wait_time = int(random.expovariate(
                            1 / common.scale))  # assign an exponentially distributed random variable to $wait_time

                    try:
                        # yield self.env.timeout(random.randrange(55, 70))
                        # yield self.env.timeout(random.randrange(45, 65))
                        random.seed(common.iteration)
                        # a = random.randint(35, 50)  # 90 82
                        # a = 1000
                        a = random.expovariate(common.lam)
                        # a = random.randint(50, 80)    # 98 94
                        # yield self.env.timeout(a)
                        yield self.env.timeout(100)
                        # yield self.env.timeout(self.wait_time)                          # new job addition will be after this wait time
                    except simpy.exceptions.Interrupt:
                        pass

                # end of while (self.generate_job):
