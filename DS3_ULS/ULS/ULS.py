import copy
import math
import random
import sys
import time
import re
from enum import Enum
from math import inf
from types import SimpleNamespace
import logging
from collections import deque, namedtuple

import networkx as nx
# 日志
import numpy as np
from matplotlib import pyplot as plt

import common
import subprocess

logger = logging.getLogger('Algorithm_RT')
ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')


class RankMetric(Enum):
    MEAN = "MEAN"
    WORST = "WORST"
    BEST = "BEST"
    EDP = "EDP"


def schedule_dag(dag, _self, communication_matrix, proc_schedules=None, relabel_nodes=True, rank_metric=RankMetric.MEAN, **kwargs):
    for i in range(len(_self.computation_matrix)):
        _self.task_schedules[i] = None

    for i in range(len(_self.communication_matrix)):
        if i not in _self.proc_schedules:
            _self.proc_schedules[i] = []

    for proc in proc_schedules:
        for schedule_event in proc_schedules[proc]:
            _self.task_schedules[schedule_event.task] = schedule_event

    with open('C:/Users/32628/Desktop/task_schedules.txt', 'w') as file1:
        file1.write("task_schedules:\n")
        for node in _self.task_schedules.keys():
            file1.write(f'{node}:{_self.task_schedules.get(node)}\n')

    # command = ['C:/Users/32628/CLionProjects/uls_1/main.exe']
    # # 执行命令
    # process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # dict_output = {}
    # with open('C:/Users/32628/Desktop/dictOutput.txt', 'r') as file:
    #     for line in file:
    #         sp = line.split()
    #         if sp[3] == 'null':
    #             dict_output[int(sp[0])] = (int(sp[1]), int(sp[2]), [], float(sp[4]), float(sp[5]), int(sp[6]))
    #         else:
    #             dict_output[int(sp[0])] = (int(sp[1]), int(sp[2]), [int(sp[3])], float(sp[4]), float(sp[5]), int(sp[6]))
    #
    # with open('C:/Users/32628/Desktop/procSchedules.txt', 'r') as file:
    #     for line in file:
    #         # 使用正则表达式匹配每一行的数据
    #         match = re.match(r'(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)', line)
    #         if match:
    #             # 将匹配的数据拆分为元组，并将其转换为整数或浮点数
    #             key = int(match.group(1))
    #             base_id = int(match.group(2))
    #             start = float(match.group(3))
    #             end = float(match.group(4))
    #             proc = int(match.group(5))
    #
    #             # 创建一个 ScheduleEvent 实例
    #             event = ScheduleEvent(base_id, start, end, proc)
    #
    #             # 如果键已经存在，追加到对应的列表中，否则创建一个新的列表
    #             if key in _self.proc_schedules:
    #                 _self.proc_schedules[key].append(event)
    #             else:
    #                 _self.proc_schedules[key] = [event]
    #
    # with open('C:/Users/32628/Desktop/taskSchedules.txt', 'r') as file:
    #     for line in file:
    #     # 使用正则表达式匹配每一行的数据
    #         match = re.match(r'(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)', line)
    #         if match:
    #             # 将匹配的数据拆分为元组，并将其转换为整数或浮点数
    #             task = int(match.group(1))
    #             base_id = int(match.group(2))
    #             start = float(match.group(3))
    #             end = float(match.group(4))
    #             proc = int(match.group(5))
    #
    #             # 检查第二个字段是否为-1，如果不是，则创建一个 SchedEvent 实例并添加到字典中
    #             if base_id != -1:
    #                 event = ScheduleEvent(task, start, end, proc)
    #                 _self.task_schedules[task] = [event]
    #
    # with open('C:/Users/32628/Desktop/texit1.txt', 'r') as file:
    #     # 逐行读取
    #     for line in file:
    #         # 移除行尾的换行符，并按空格分割
    #         parts = line.strip().split()
    #         common.texit[parts[0]] = parts[1]

    # 划分子截止时间
    # 给dag中各node设置好pr、psd、avgweight参数

    _divide_sub_deadline(_self, dag)

    # 按sd大小非递减排序
    sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['sd'])

    with open("C:/Users/32628/Desktop/sorted_nodes.txt", 'w') as file:
        file.write(str(sorted_nodes) + '\n')
    if sorted_nodes[0] != _self.root_node:
        idx = sorted_nodes.index(_self.root_node)
        for i in range(idx, 0, -1):
            sorted_nodes[i] = sorted_nodes[i - 1]
        sorted_nodes[0] = _self.root_node
    for node in _self.over_time:
        # proc_num, idx, [], task.end, task.start, task.task
        minTaskSchedule = ScheduleEvent(node, common.table[node][4], common.table[node][3], common.table[node][0])
        _self.task_schedules[node] = minTaskSchedule
        _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
        _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc],
                                                            key=lambda schedule_event: (
                                                                schedule_event.end, schedule_event.start))

    # 处理器选择
    for node in sorted_nodes:
        if node in _self.over_time:
            continue
        # 当前没进入该分支
        if node == _self.root_node or node == _self.terminal_node:
            continue

        # node结点的分配情况
        minTaskSchedule = ScheduleEvent(node, inf, inf, -1)
        for proc in range(len(communication_matrix)):
            if _self.computation_matrix[node, proc] == inf:
                continue
            taskschedule = EFT(_self, dag, node, proc)
            if taskschedule.end < minTaskSchedule.end:
                minTaskSchedule = taskschedule

        # 将结果存入全局变量
        _self.task_schedules[node] = minTaskSchedule
        _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
        _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc],
                                                            key=lambda schedule_event: (
                                                                schedule_event.end, schedule_event.start))

    dict_output = {}
    for proc_num, proc_tasks in _self.proc_schedules.items():
        # idx:对应processor要执行任务序列的每一个index
        # task:对应每个任务
        for idx, task in enumerate(proc_tasks):
            if idx > 0 and (proc_tasks[idx - 1].end - proc_tasks[idx - 1].start > 0):
                dict_output[task.task] = (proc_num, idx, [proc_tasks[idx - 1].task], task.end, task.start, task.task)
            else:
                dict_output[task.task] = (proc_num, idx, [], task.end, task.start, task.task)

    return _self.proc_schedules, _self.task_schedules, dict_output


def EFT(_self, dag, node, proc):
    # ready_time初始化为当前时间偏移，如果node没有前驱节点就为该值
    ready_time = _self.time_offset
    # 在当前dag中遍历前驱节点
    for prednode in list(dag.predecessors(node)):
        if prednode == _self.root_node:
            continue
        predjob = _self.task_schedules[prednode]
        # 如果前驱结点到node的通信成本为0，就绪时间为前驱节点的结束时间
        # if predjob is None:
        #     print(node, prednode)
        if predjob.proc == proc:
            ready_time_t = predjob.end
        # 如果当前processor和前驱节点所用processor的communication_cost!=0,则当前节点的就绪时间为该前驱节点结束时间+数据量/传输速率
        else:
            ready_time_t = predjob.end + dag[predjob.task][node]['weight'] / _self.communication_matrix[
                predjob.proc, proc]
        # 找到最晚就绪时间
        if ready_time_t > ready_time:
            ready_time = ready_time_t
    # 如果是根结点就没有前驱结点
    if node is not _self.root_node:
        # 找到当前的task
        curr_task = getTask(_self, node)
        if curr_task.jobID != _self.jobID:
            offset = node - curr_task.base_ID
            # print(offset)
            # 前驱结点的数组
            prednode_1 = curr_task.predecessors

            # 遍历前驱数组
            for prednode in list(prednode_1):
                if prednode in dag.nodes():
                    continue
                if common.table != -1:
                    # 如果前驱结点已经被调度
                    if prednode in common.table.keys():
                        # print('pre: ',prednode)
                        predjob = common.table[prednode]
                        if predjob[0] == proc:
                            ready_time_t = predjob[3]
                        # 如果当前processor和前驱节点所用processor的communication_cost!=0,则当前节点的就绪时间为该前驱节点结束时间+数据量/传输速率
                        elif proc != common.table[curr_task.ID][0] and getTask_1(_self, prednode) != -1:
                            ready_time_t = _self.time_offset + \
                                           common.communication_dict[curr_task.jobID][predjob[5] - offset][
                                               curr_task.base_ID] / _self.communication_matrix[
                                               predjob[0], proc]
                        else:
                            ready_time_t = predjob[3] + \
                                           common.communication_dict[curr_task.jobID][predjob[5] - offset][
                                               curr_task.base_ID] / _self.communication_matrix[
                                               predjob[0], proc]
                        # 找到最早就绪时间
                        if ready_time_t > ready_time:
                            ready_time = ready_time_t
    # 处理器执行时间
    computation_time = _self.computation_matrix[node, proc]

    # 该处理器上执行的所有任务list
    job_list = _self.proc_schedules[proc]

    # 遍历这些任务
    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        # insertion-based scheduling policy
        # 该processor的任务执行队列中第一个任务
        if idx == 0:
            # 该processor中第一个节点的开始时间大于当前节点的就绪时间+执行时间,即当前节点执行不影响前一个节点
            if (prev_job.start - computation_time) - ready_time > 0:
                job_start = ready_time
                min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
                break

        # 该processor的任务执行队列中最后一个任务
        if idx == len(job_list) - 1:
            job_start = max(ready_time, prev_job.end)
            # EFT = job_start + computation_time
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
        next_job = job_list[idx + 1]
        if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
    # 当前processor执行队列为空,即可立即使用,available[j]=0
    else:
        min_schedule = ScheduleEvent(node, ready_time, ready_time + computation_time, proc)

    return min_schedule


def _divide_sub_deadline(_self, dag):
    deadline = _self.deadline
    common.deadline = deadline
    nx.set_node_attributes(dag, {_self.root_node: findAvg(_self.computation_matrix, _self.root_node)}, "t")
    nx.set_node_attributes(dag, {_self.root_node: 0}, "t1")
    nx.set_node_attributes(dag, {_self.root_node: 0}, "visit")
    visit_queue = deque(dag.successors(_self.root_node))
    while visit_queue:
        node = visit_queue.pop()
        # 如果存在未得到t或者未访问过的父结点
        while _node_can_be_processed(_self, dag, node) is not True:
            try:
                node2 = visit_queue.pop()
            except IndexError:
                raise RuntimeError(
                    f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            visit_queue.appendleft(node)
            node = node2
        nx.set_node_attributes(dag, {node: 0}, "visit")
        temp_1 = 0
        # 旧工作流
        if 'sd' in dag.nodes()[node]:
            # 如果旧工作流超时，不再计算
            if node in _self.over_time:
                visit_queue.extendleft([succnode for succnode in dag.successors(node) if succnode not in visit_queue])
                continue
            EST = findReadyTime(_self, node, dag)
            DRT = max(EST - _self.time_offset, 0)
            maxTemp = 0
            for prednode in dag.predecessors(node):
                temp = dag.nodes()[prednode]['t'] + dag[prednode][node]['avgweight']
                if temp > maxTemp:
                    maxTemp = temp
            temp_1 = max(maxTemp, DRT)
        # 新工作流
        else:
            for prednode in dag.predecessors(node):
                temp = dag.nodes()[prednode]['t'] + dag[prednode][node]['avgweight']
                if temp > temp_1:
                    temp_1 = temp
        nx.set_node_attributes(dag, {node: temp_1}, "t1")
        nx.set_node_attributes(dag, {node: temp_1 + findAvg(_self.computation_matrix, node)}, "t")
        # if node == _self.terminal_node:
        #     common.texit[_self.jobID] = dag.nodes()[node]['t']
        if node in dag.predecessors(_self.terminal_node):
            if 'sd' not in dag.nodes()[node]:
                if _self.jobID in common.texit:
                    common.texit[_self.jobID] = max(dag.nodes()[node]['t'], common.texit[_self.jobID])
                else:
                    common.texit[_self.jobID] = dag.nodes()[node]['t']
            else:
                # if task.jobID in common.texit:
                #     common.texit[task.jobID] = max(dag.nodes()[node]['t'], common.texit[task.jobID])
                # else:
                #     common.texit[task.jobID] = dag.nodes()[node]['t']
                # if dag.nodes()[node]['t'] > common.texit[task.jobID]:
                    # print(1)
                pass
        visit_queue.extendleft([succnode for succnode in dag.successors(node) if succnode not in visit_queue])
    for node in dag.nodes():
        del dag.nodes()[node]['visit']
        # 新工作流
        if 'sd' not in dag.nodes()[node]:
            nx.set_node_attributes(dag, {
                node: dag.nodes()[node]['t'] / common.texit[_self.jobID] * _self.deadline}, "sd")
            # if dag.nodes()[node]['t1'] == 0:
            #     dag.nodes()[node]['sd'] = 0
            # else:
            #     dag.nodes()[node]['sd'] = 2 / ((1 / dag.nodes()[node]['sd']) + (1 / dag.nodes()[node]['t1']))
            dag.nodes()[node]['sd'] = (dag.nodes()[node]['sd'] + dag.nodes()[node]['t1']) / 2
        else:
            task = getTask(_self, node)
            # 如果所在的工作流已经超时，排序向后调整
            if common.deadline_dict[task.jobID] <= _self.time_offset:
                continue
            else:
                nx.set_node_attributes(dag, {
                    node: (dag.nodes()[node]['t'] / common.texit[task.jobID] * (
                            common.deadline_dict[task.jobID] - _self.time_offset))}, "sd")
                # if dag.nodes()[node]['t1'] == 0:
                #     dag.nodes()[node]['sd'] = 0
                # else:
                #     dag.nodes()[node]['sd'] = 2 / ((1 / dag.nodes()[node]['sd']) + (1 / dag.nodes()[node]['t1']))
                dag.nodes()[node]['sd'] = (dag.nodes()[node]['sd'] + dag.nodes()[node]['t1']) / 2


def getTask(_self, node):
    if node in _self.dict1:
        return _self.dict1[node]
    return -1


def getTask_1(_self, node):
    if node in _self.dict2:
        return _self.dict2[node]
    return -1


def getTask_10(node):
    for task_1 in common.TaskQueues.completed.list:
        if task_1.ID == node:
            return task_1
    return -1


def _node_can_be_processed(_self, dag, node):
    for prednode in dag.predecessors(node):
        if 't' not in dag.nodes()[prednode] or 'visit' not in dag.nodes()[prednode]:
            return False
    return True


def findMin(computation, index):
    temp = inf
    for i in range(len(computation[index])):
        if computation[index][i] != inf and computation[index][i] < temp:
            temp = computation[index][i]
    return temp


def findAvg(computation, index):
    sum = 0
    num = 0
    for i in range(len(computation[index])):
        if computation[index][i] != inf:
            sum += computation[index][i]
            num += 1
    return sum / num


def findAvg_1(computation, index, proc):
    sum = 0
    num = 0
    for i in range(len(computation[index])):
        if i == proc:
            continue
        if computation[index][i] != inf:
            sum += computation[index][i]
            num += 1
    return sum / num


def findReadyTime(_self, node, dag):
    # ready_time初始化为当前时间偏移，如果node没有前驱节点就为该值
    if node is not _self.root_node:
        curr_task = getTask(_self, node)
        # print(curr_task)
        offset = curr_task.ID - curr_task.base_ID
        # 前驱结点的数组
        prednode_1 = curr_task.predecessors
        EST = 0
        avgCommunicationCost = _self.avgComm
        t1 = 0
        job = common.table[node]
        for pred in list(prednode_1):
            predjob_1 = common.table[pred]
            a = predjob_1[3]
            b = common.communication_dict[curr_task.jobID][predjob_1[5] - offset][curr_task.base_ID]
            c = _self.communication_matrix[predjob_1[0], job[0]]
            d = _self.time_offset
            e = a + b / c - d
            tmp = max(0, predjob_1[3] + common.communication_dict[curr_task.jobID][predjob_1[5] - offset][
                curr_task.base_ID] / _self.communication_matrix[predjob_1[0], job[0]] - _self.time_offset)
            t1 = max(t1, tmp)
        for prednode in list(prednode_1):
            if prednode not in common.running:
                continue
            predjob = common.table[prednode]
            if getTask_1(_self, prednode) != -1:
                avgCommunicationCost = _self.avgComm
                # t1 = 0
                # for pred in list(prednode_1):
                #     predjob_1 = common.table[pred]
                #     tmp = max(0, predjob_1[3] + common.communication_dict[curr_task.jobID][predjob_1[5] - offset][
                #         curr_task.base_ID] / _self.communication_matrix[predjob_1[0], job[0]] - _self.time_offset)
                #     t1 = max(t1, tmp)
                sum = (_self.computation_matrix[node][job[0]] + t1) + (
                            findAvg_1(_self.computation_matrix, node, job[0]) + (
                                common.communication_dict[curr_task.jobID][prednode - offset][
                                    node - offset] / avgCommunicationCost))
                temp = (findAvg_1(_self.computation_matrix, node, job[0]) + (
                            common.communication_dict[curr_task.jobID][prednode - offset][
                                node - offset] / avgCommunicationCost)) / sum * (
                               _self.time_offset + common.communication_dict[curr_task.jobID][prednode - offset][
                           node - offset] / _self.avgComm) + (_self.computation_matrix[node][job[0]] + t1) / sum * (
                               predjob[3] + common.communication_dict[curr_task.jobID][prednode - offset][
                           node - offset] / _self.avgComm)

            else:
                temp = predjob[3] + common.communication_dict[curr_task.jobID][prednode - offset][
                    node - offset] / _self.avgComm
            if temp > EST:
                EST = temp
        return EST
