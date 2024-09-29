"""Core code to be used for scheduling a task DAG with HEFT"""
import time
from collections import deque, namedtuple
from math import inf

import common
from heft.gantt import showGanttChart
from types import SimpleNamespace
from enum import Enum

import argparse
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger('heft')

ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')

"""
Default computation matrix - taken from Topcuoglu 2002 HEFT paper
computation matrix: v x q matrix with v tasks and q PEs
"""
W0 = np.array([
    [14, 16, 9],
    [13, 19, 18],
    [11, 13, 19],
    [13, 8, 17],
    [12, 13, 10],
    [13, 16, 9],
    [7, 15, 11],
    [5, 11, 14],
    [18, 12, 20],
    [21, 7, 16]
])

"""
Default communication matrix - not listed in Topcuoglu 2002 HEFT paper
communication matrix: q x q matrix with q PEs

Note that a communication cost of 0 is used for a given processor to itself
"""
C0 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])


class RankMetric(Enum):
    MEAN = "MEAN"
    WORST = "WORST"
    BEST = "BEST"
    EDP = "EDP"


class OpMode(Enum):
    EFT = "EFT"
    EDP_REL = "EDP RELATIVE"
    EDP_ABS = "EDP ABSOLUTE"
    ENERGY = "ENERGY"


def schedule_dag(name, jobID, dag, computation_matrix=W0, communication_matrix=C0, proc_schedules=None, time_offset=0,
                 relabel_nodes=True, rank_metric=RankMetric.MEAN, **kwargs):
    """
    Given an application DAG and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the HEFT schedule
    of that DAG onto that set of PEs
    """
    # print(dag.nodes())
    # print(proc_schedules)
    if proc_schedules == None:
        proc_schedules = {}

    _self = {
        'computation_matrix': computation_matrix,
        'communication_matrix': communication_matrix,
        'task_schedules': {},
        'proc_schedules': proc_schedules,
        'numExistingJobs': 0,
        'time_offset': time_offset,
        'root_node': None
    }
    _self = SimpleNamespace(**_self)
    # print(computation_matrix)
    for proc in proc_schedules:
        _self.numExistingJobs = _self.numExistingJobs + len(proc_schedules[proc])

    if relabel_nodes:
        dag = nx.relabel_nodes(dag, dict(map(lambda node: (node, node + _self.numExistingJobs), list(dag.nodes()))))
    else:
        # Negates any offsets that would have been needed had the jobs been relabeled
        _self.numExistingJobs = 0

    for i in range(_self.numExistingJobs + len(_self.computation_matrix)):
        _self.task_schedules[i] = None
    for i in range(len(_self.communication_matrix)):
        if i not in _self.proc_schedules:
            _self.proc_schedules[i] = []

    # print(proc_schedules)
    for proc in proc_schedules:
        for schedule_event in proc_schedules[proc]:
            _self.task_schedules[schedule_event.task] = schedule_event
    # print(_self.proc_schedules)
    # print(_self.task_schedules)
    # print()

    # Nodes with no successors cause the any expression to be empty
    root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
    root_node = root_node[0]
    _self.root_node = root_node
    start_time = int(round(time.time() * 1000))
    _compute_ranku(_self, dag, metric=rank_metric, **kwargs)

    sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['ranku'], reverse=True)
    if sorted_nodes[0] != root_node:
        idx = sorted_nodes.index(root_node)
        sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]
    late = 0
    for node in sorted_nodes:
        if _self.task_schedules[node] is not None:
            continue
        minTaskSchedule = ScheduleEvent(node, inf, inf, -1)
        minEDP = inf
        op_mode = kwargs.get("op_mode", OpMode.EFT)
        if op_mode == OpMode.EDP_ABS:
            taskschedules = []
            minScheduleStart = inf

            for proc in range(len(communication_matrix)):
                taskschedule = _compute_eft(_self, dag, node, proc)
                edp_t = ((taskschedule.end - taskschedule.start) ** 2) * kwargs["power_dict"][node][proc]
                if (edp_t < minEDP):
                    minEDP = edp_t
                    minTaskSchedule = taskschedule
                elif (edp_t == minEDP and taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule

        elif op_mode == OpMode.EDP_REL:
            taskschedules = []
            minScheduleStart = inf

            for proc in range(len(communication_matrix)):
                taskschedules.append(_compute_eft(_self, dag, node, proc))
                if taskschedules[proc].start < minScheduleStart:
                    minScheduleStart = taskschedules[proc].start

            for taskschedule in taskschedules:
                # Use the makespan relative to the earliest potential assignment to encourage load balancing
                edp_t = ((taskschedule.end - minScheduleStart) ** 2) * kwargs["power_dict"][node][taskschedule.proc]
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
                taskschedule = _compute_eft(_self, dag, node, proc)
                if (taskschedule.end < minTaskSchedule.end):
                    minTaskSchedule = taskschedule
        late = max(minTaskSchedule.end, late)
        terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
        common.end.append(terminal_node[0])
        if node == terminal_node[0]:
            # print(minTaskSchedule.end)
            if minTaskSchedule.end > common.deadline_dict[jobID]:
                common.num_of_out_1 += 1
                common.overtime_sum += minTaskSchedule.end - common.deadline_dict[jobID]
                print('-----------------------')
                print('ID', 'job', name, ' is out of deadline')
                print('Arrive Time:', common.arrive_time[jobID],
                      ' Actual Finish Time: ', minTaskSchedule.end, '   Deadline:',
                      common.deadline_dict[jobID])
                print('-----------------------')
            else:
                print('ID', jobID, name, 'Arrive Time:', common.arrive_time[jobID],
                      ' Actual Finish Time: ', minTaskSchedule.end, '   Deadline:',
                      common.deadline_dict[jobID])
        _self.task_schedules[node] = minTaskSchedule
        _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
        _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc], key=lambda schedule_event: (schedule_event.end, schedule_event.start))

    end_time = int(round(time.time() * 1000))
    common.exe_time += end_time - start_time

    dict_output = {}
    for proc_num, proc_tasks in _self.proc_schedules.items():
        for idx, task in enumerate(proc_tasks):
            if idx > 0 and (proc_tasks[idx - 1].end - proc_tasks[idx - 1].start > 0):
                dict_output[task.task] = (proc_num, idx, [proc_tasks[idx - 1].task])
            else:
                dict_output[task.task] = (proc_num, idx, [])
    common.running_task = _self.proc_schedules
    return _self.proc_schedules, _self.task_schedules, dict_output


def _scale_by_operating_freq(_self, **kwargs):
    if "operating_freqs" not in kwargs:
        logger.debug("No operating frequency argument is present, assuming at max frequency and values are unchanged")
        return
    return  # TODO
    # for pe_num, freq in enumerate(kwargs["operating_freqs"]):
    # _self.computation_matrix[:, pe_num] = _self.computation_matrix[:, pe_num] * (1 + compute_DVFS_performance_slowdown(pe_num, freq))


def _compute_ranku(_self, dag, metric=RankMetric.MEAN, **kwargs):
    """
    Uses a basic BFS approach to traverse upwards through the graph assigning ranku along the way
    """
    terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    terminal_node = terminal_node[0]

    # TODO: Should this be configurable?
    # avgCommunicationCost = np.mean(_self.communication_matrix[np.where(_self.communication_matrix > 0)])
    diagonal_mask = np.ones(_self.communication_matrix.shape, dtype=bool)
    np.fill_diagonal(diagonal_mask, 0)
    avgCommunicationCost = np.mean(_self.communication_matrix[diagonal_mask])
    for edge in dag.edges():
        nx.set_edge_attributes(dag, {edge: float(dag.get_edge_data(*edge)['weight']) / avgCommunicationCost},
                               'avgweight')

    # Utilize a masked array so that np.mean, etc, calculations ignore the entries that are inf
    comp_matrix_masked = np.ma.masked_invalid(_self.computation_matrix)

    nx.set_node_attributes(dag, {terminal_node: np.mean(comp_matrix_masked[terminal_node - _self.numExistingJobs])},
                           "ranku")
    visit_queue = deque(dag.predecessors(terminal_node))

    while visit_queue:
        node = visit_queue.pop()
        while _node_can_be_processed(_self, dag, node) is not True:
            try:
                node2 = visit_queue.pop()
            except IndexError:
                raise RuntimeError(
                    f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            visit_queue.appendleft(node)
            node = node2

        if metric == RankMetric.MEAN:
            max_successor_ranku = -1
            for succnode in dag.successors(node):
                val = float(dag[node][succnode]['avgweight']) + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            nx.set_node_attributes(dag, {
                node: np.mean(comp_matrix_masked[node - _self.numExistingJobs]) + max_successor_ranku}, "ranku")

        elif metric == RankMetric.WORST:
            max_successor_ranku = -1
            max_node_idx = np.where(comp_matrix_masked[node - _self.numExistingJobs] == max(
                comp_matrix_masked[node - _self.numExistingJobs]))[0][0]
            for succnode in dag.successors(node):
                max_succ_idx = np.where(comp_matrix_masked[succnode - _self.numExistingJobs] == max(
                    comp_matrix_masked[succnode - _self.numExistingJobs]))[0][0]
                val = _self.communication_matrix[max_node_idx, max_succ_idx] + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            nx.set_node_attributes(dag, {
                node: comp_matrix_masked[node - _self.numExistingJobs, max_node_idx] + max_successor_ranku}, "ranku")

        elif metric == RankMetric.BEST:
            min_successor_ranku = inf
            min_node_idx = np.where(comp_matrix_masked[node - _self.numExistingJobs] == min(
                comp_matrix_masked[node - _self.numExistingJobs]))[0][0]
            for succnode in dag.successors(node):
                min_succ_idx = np.where(comp_matrix_masked[succnode - _self.numExistingJobs] == min(
                    comp_matrix_masked[succnode - _self.numExistingJobs]))[0][0]
                val = _self.communication_matrix[min_node_idx, min_succ_idx] + dag.nodes()[succnode]['ranku']
                if val < min_successor_ranku:
                    min_successor_ranku = val
            nx.set_node_attributes(dag, {
                node: comp_matrix_masked[node - _self.numExistingJobs, min_node_idx] + min_successor_ranku}, "ranku")

        elif metric == RankMetric.EDP:
            power_dict = kwargs.get("power_dict", np.array([[]]))
            power_dict_masked = np.ma.masked_invalid(power_dict[node])
            max_successor_ranku = -1
            for succnode in dag.successors(node):
                val = float(dag[node][succnode]['avgweight']) + dag.nodes()[succnode]['ranku']
                if val > max_successor_ranku:
                    max_successor_ranku = val
            avg_edp = np.mean(comp_matrix_masked[node - _self.numExistingJobs]) ** 2 * np.mean(power_dict_masked)
            nx.set_node_attributes(dag, {node: avg_edp + max_successor_ranku}, "ranku")

        else:
            raise RuntimeError(f"Unrecognied Rank-U metric {metric}, unable to compute upward rank")

        visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])


def _node_can_be_processed(_self, dag, node):
    """
    Validates that a node is able to be processed in Rank U calculations. Namely, that all of its successors have their Rank U values properly assigned
    Otherwise, errors can occur in processing DAGs of the form
    A
    |\
    | B
    |/
    C
    Where C enqueues A and B, A is popped off, and it is unable to be processed because B's Rank U has not been computed
    """
    for succnode in dag.successors(node):
        if 'ranku' not in dag.nodes()[succnode]:
            return False
    return True


def _compute_eft(_self, dag, node, proc):
    """
    Computes the EFT of a particular node if it were scheduled on a particular processor
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
    It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
    """
    ready_time = _self.time_offset
    for prednode in list(dag.predecessors(node)):
        predjob = _self.task_schedules[prednode]
        if _self.communication_matrix[predjob.proc, proc] == 0 or predjob.proc == proc:
            ready_time_t = predjob.end
        else:
            ready_time_t = predjob.end + dag[predjob.task][node]['weight'] / _self.communication_matrix[predjob.proc, proc]
        if ready_time_t > ready_time:
            ready_time = ready_time_t

    computation_time = _self.computation_matrix[node - _self.numExistingJobs, proc]
    job_list = _self.proc_schedules[proc]
    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        if idx == 0:
            if (prev_job.start - computation_time) - ready_time > 0:
                job_start = ready_time
                min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
                break
        if idx == len(job_list) - 1:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
        next_job = job_list[idx + 1]
        # Start of next job - computation time == latest we can start in this window
        # Max(ready_time, previous job's end) == earliest we can start in this window
        # If there's space in there, schedule in it
        if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
    else:
        # For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this processor are 0
        min_schedule = ScheduleEvent(node, ready_time, ready_time + computation_time, proc)
    return min_schedule


def readCsvToNumpyMatrix(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column,
    this function reads that data into a numpy matrix and strips the top row and leftmost column
    """
    with open(csv_file) as fd:
        logger.debug(f"Reading the contents of {csv_file} into a matrix")
        contents = fd.read()
        contentsList = contents.split('\n')
        contentsList = list(map(lambda line: line.split(','), contentsList))
        contentsList = contentsList[0:len(contentsList) - 1] if contentsList[len(contentsList) - 1] == [
            ''] else contentsList

        matrix = np.array(contentsList)
        matrix = np.delete(matrix, 0, 0)  # delete the first row (entry 0 along axis 0)
        matrix = np.delete(matrix, 0, 1)  # delete the first column (entry 0 along axis 1)
        matrix = matrix.astype(float)
        logger.debug(f"After deleting the first row and column of input data, we are left with this matrix:\n{matrix}")
        return matrix


def readCsvToDict(csv_file):
    """
    Given an input file consisting of a comma separated list of numeric values with a single header row and header column,
    this function reads that data into a dictionary with keys that are node numbers and values that are the CSV lists
    """
    with open(csv_file) as fd:
        matrix = readCsvToNumpyMatrix(csv_file)

        outputDict = {}
        for row_num, row in enumerate(matrix):
            outputDict[row_num] = row
        return outputDict


def readDagMatrix(dag_file, show_dag=False):
    """
    Given an input file consisting of a connectivity matrix, reads and parses it into a networkx Directional Graph (DiGraph)
    """
    matrix = readCsvToNumpyMatrix(dag_file)

    dag = nx.DiGraph(matrix)
    dag.remove_edges_from(
        # Remove all edges with weight of 0 since we have no placeholder for "this edge doesn't exist" in the input file
        [edge for edge in dag.edges() if dag.get_edge_data(*edge)['weight'] == '0.0']
    )

    if show_dag:
        nx.draw(dag, pos=nx.nx_pydot.graphviz_layout(dag, prog='dot'), with_labels=True)
        plt.show()

    return dag


def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for finding HEFT schedules for given DAG task graphs")
    parser.add_argument("-d", "--dag_file",
                        help="File containing input DAG to be scheduled. Uses default 10 node dag from Topcuoglu 2002 if none given.",
                        type=str, default="test/canonicalgraph_task_connectivity.csv")
    parser.add_argument("-p", "--pe_connectivity_file",
                        help="File containing connectivity/bandwidth information about PEs. Uses a default 3x3 matrix from Topcuoglu 2002 if none given.",
                        type=str, default="test/canonicalgraph_resource_BW.csv")
    parser.add_argument("-t", "--task_execution_file",
                        help="File containing execution times of each task on each particular PE. Uses a default 10x3 matrix from Topcuoglu 2002 if none given.",
                        type=str, default="test/canonicalgraph_task_exe_time.csv")
    parser.add_argument("-l", "--loglevel",
                        help="The log level to be used in this module. Default: INFO",
                        type=str, default="INFO", dest="loglevel",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--metric",
                        help="Specify which metric to use when performing upward rank calculation",
                        type=RankMetric, default=RankMetric.MEAN, dest="rank_metric", choices=list(RankMetric))
    parser.add_argument("--showDAG",
                        help="Switch used to enable display of the incoming task DAG",
                        dest="showDAG", action="store_true")
    parser.add_argument("--showGantt",
                        help="Switch used to enable display of the final scheduled Gantt chart",
                        dest="showGantt", action="store_true")
    return parser


if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    logger.setLevel(logging.getLevelName(args.loglevel))
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.getLevelName(args.loglevel))
    consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
    logger.addHandler(consolehandler)

    communication_matrix = readCsvToNumpyMatrix(args.pe_connectivity_file)
    computation_matrix = readCsvToNumpyMatrix(args.task_execution_file)
    dag = readDagMatrix(args.dag_file, args.showDAG)

    processor_schedules, _, _ = schedule_dag(dag, communication_matrix=communication_matrix,
                                             computation_matrix=computation_matrix, rank_metric=args.rank_metric)
    for proc, jobs in processor_schedules.items():
        logger.info(f"Processor {proc} has the following jobs:")
        logger.info(f"\t{jobs}")
    if args.showGantt:
        showGanttChart(processor_schedules)
