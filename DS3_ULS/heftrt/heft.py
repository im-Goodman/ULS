from collections import deque, namedtuple
from math import inf
from heft.gantt import showGanttChart
from types import SimpleNamespace
from enum import Enum

import argparse
import sys
import common
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')

class RankMetric(Enum):
    MEAN = "MEAN"
    EDP = "EDP"

class OpMode(Enum):
    EFT = "EFT"
    EDP_REL = "EDP RELATIVE"
    EDP_ABS = "EDP ABSOLUTE"
    ENERGY = "ENERGY"

def schedule_queue(ready_queue, computation_dict=None, communication_matrix=None, proc_schedules=None, time_offset=0, rank_metric=RankMetric.MEAN, **kwargs):
    """
    Given a ready queue and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the HEFT schedule
    of that ready queue onto that set of PEs
    """
    common.times += 1
    _self = {
        'computation_dict': computation_dict,
        'communication_matrix': communication_matrix,
        'task_schedules': {},
        'proc_schedules': proc_schedules,
        'time_offset': time_offset
    }

    _self = SimpleNamespace(**_self)
    for task in ready_queue:
        # print(task.ID, end=" ")
        _self.task_schedules[task.ID] = None
    # print()
    # print(kwargs["power_dict"])
    for task in ready_queue:
        _self.task_schedules[task.ID] = None
    for i in range(len(_self.communication_matrix)):
        if i not in _self.proc_schedules:
            _self.proc_schedules[i] = []

    for proc in proc_schedules:
        for schedule_event in proc_schedules[proc]:
            _self.task_schedules[schedule_event.task] = schedule_event

    _compute_ranku(_self, ready_queue, metric=rank_metric, **kwargs)

    sorted_nodes = sorted(ready_queue, key=lambda task: task.ranku, reverse=True)
    for node in sorted_nodes:
        if _self.task_schedules[node.ID] is not None:
            continue
        minTaskSchedule = ScheduleEvent(node.ID, inf, inf, -1)
        minEDP = inf
        op_mode = kwargs.get("op_mode", OpMode.EFT)
        if op_mode == OpMode.EDP_ABS:
            assert "power_dict" in kwargs, "In order to perform EDP-based processor assignment, a power_dict is required"
            taskschedules = []
            minScheduleStart = inf

            for proc in range(len(communication_matrix)):
                taskschedule = _compute_eft(_self, node, proc)
                edp_t = ((taskschedule.end - taskschedule.start)**2) * kwargs["power_dict"][node.ID][proc]
                if (edp_t < minEDP):
                    minEDP = edp_t
                    minTaskSchedule = taskschedule
                elif (edp_t == minEDP and taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule

        elif op_mode == OpMode.EDP_REL:
            assert "power_dict" in kwargs, "In order to perform EDP-based processor assignment, a power_dict is required"
            taskschedules = []
            minScheduleStart = inf

            for proc in range(len(communication_matrix)):
                taskschedules.append(_compute_eft(_self, node, proc))
                if taskschedules[proc].start < minScheduleStart:
                    minScheduleStart = taskschedules[proc].start

            for taskschedule in taskschedules:
                # Use the makespan relative to the earliest potential assignment to encourage load balancing
                edp_t = ((taskschedule.end - minScheduleStart)**2) * kwargs["power_dict"][node.ID][taskschedule.proc]
                if (edp_t < minEDP):
                    minEDP = edp_t
                    minTaskSchedule = taskschedule
                elif (edp_t == minEDP and taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule

        elif op_mode == OpMode.ENERGY:
            assert False, "Feature not implemented"
            assert "power_dict" in kwargs, "In order to perform Energy-based processor assignment, a power_dict is required"

        else:
            for proc in range(len(communication_matrix)):
                taskschedule = _compute_eft(_self, node, proc)
                if (taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule

        _self.task_schedules[node.ID] = minTaskSchedule
        _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
        _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc], key=lambda schedule_event: (schedule_event.end, schedule_event.start))

    for proc in range(len(_self.proc_schedules)):
        for job in range(len(_self.proc_schedules[proc])-1):
            first_job = _self.proc_schedules[proc][job]
            second_job = _self.proc_schedules[proc][job+1]
            assert first_job.end <= second_job.start, \
            f"Jobs on a particular processor must finish before the next can begin, but job {first_job.task} on processor {first_job.proc} ends at {first_job.end} and its successor {second_job.task} starts at {second_job.start}"

    dict_output = {}
    for proc_num, proc_tasks in _self.proc_schedules.items():
        for idx, task in enumerate(proc_tasks):
            if idx > 0 and (proc_tasks[idx-1].end - proc_tasks[idx-1].start > 0):
                dict_output[task.task] = (proc_num, idx, [proc_tasks[idx-1].task])
            else:
                dict_output[task.task] = (proc_num, idx, [])
    # print()
    return dict_output

def _compute_ranku(_self, ready_queue, metric=RankMetric.MEAN, **kwargs):
    """
    Just sort the ready queue by each node's average computation cost
    """

    if metric == RankMetric.MEAN:
        for task in ready_queue:
            # Utilize a masked array so that np.mean, etc, calculations ignore the entries that are inf
            comp_matrix_masked = np.ma.masked_invalid(_self.computation_dict[task.ID])
            task.ranku = np.mean(comp_matrix_masked)
    elif metric == RankMetric.EDP:
        assert "power_dict" in kwargs, "In order to perform EDP-based Rank Method, a power_dict is required"
        power_dict = kwargs.get("power_dict")
        for task in ready_queue:
            # Utilize masked arrays so that np.mean, etc, calculations ignore the entries that are inf
            comp_matrix_masked = np.ma.masked_invalid(_self.computation_dict[task.ID])
            power_dict_masked = np.ma.masked_invalid(power_dict[task.ID])
            task.ranku = np.mean(comp_matrix_masked)**2 * np.mean(power_dict_masked)
    else:
        raise RuntimeError(f"Unrecognied Rank-U metric {metric}, unable to compute upward rank")

def _compute_eft(_self, node, proc):
    """
    Computes the EFT of a particular node if it were scheduled on a particular processor
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
    It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
    """
    ready_time = _self.time_offset
    computation_time = _self.computation_dict[node.ID][proc]
    job_list = _self.proc_schedules[proc]

    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        if idx == 0:
            if (prev_job.start - computation_time) - ready_time > 0:
                job_start = ready_time
                min_schedule = ScheduleEvent(node.ID, job_start, job_start+computation_time, proc)
                break
        if idx == len(job_list)-1:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node.ID, job_start, job_start + computation_time, proc)
            break
        next_job = job_list[idx+1]
        #Start of next job - computation time == latest we can start in this window
        #Max(ready_time, previous job's end) == earliest we can start in this window
        #If there's space in there, schedule in it
        if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node.ID, job_start, job_start + computation_time, proc)
            break
    else:
        #For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this processor are 0
        min_schedule = ScheduleEvent(node.ID, ready_time, ready_time + computation_time, proc)
    return min_schedule
