"""Core code to be used for scheduling a task DAG with HEFT"""
import time
from collections import deque, namedtuple
from math import inf

import networkx

import common
from peft.gantt import showGanttChart
from types import SimpleNamespace

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

logger = logging.getLogger('peft')

logLevel = "ERROR"

logger.setLevel(logging.getLevelName(logLevel))
consolehandler = logging.StreamHandler()
consolehandler.setLevel(logging.getLevelName(logLevel))
consolehandler.setFormatter(logging.Formatter("%(levelname)8s : %(name)16s : %(message)s"))
logger.addHandler(consolehandler)

ScheduleEvent = namedtuple('ScheduleEvent', 'task start end proc')

"""
Default computation matrix - taken from Arabnejad 2014 PEFT paper
computation matrix: v x q matrix with v tasks and q PEs
"""
W0 = np.array([
    [22, 21, 36],
    [22, 18, 18],
    [32, 27, 19],
    [7, 10, 17],
    [29, 27, 10],
    [26, 17, 9],
    [14, 25, 11],
    [29, 23, 14],
    [15, 21, 20],
    [13, 16, 16]
])

"""
Default communication matrix - not listed in Arabnejad 2014 PEFT paper
communication matrix: q x q matrix with q PEs

Note that a communication cost of 0 is used for a given processor to itself
"""
C0 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# computation_matrix=computation_matrix,
# communication_matrix=common.ResourceManager.comm_band,
# proc_schedules=running_tasks,
# time_offset=self.env.now,
# relabel_nodes=False
def schedule_dag(dag, PES, computation_matrix=W0, communication_matrix=C0, proc_schedules=None, time_offset=0, ready_queue=[], relabel_nodes=True):
    """
    Given an application DAG and a set of matrices specifying PE bandwidth and (task, pe) execution times, computes the HEFT schedule
    of that DAG onto that set of PEs 
    """
    if proc_schedules == None:
        proc_schedules = {}

    _self = {
        'computation_matrix': computation_matrix,
        'communication_matrix': communication_matrix,
        'task_schedules': {},
        'proc_schedules': proc_schedules,
        'numExistingJobs': 0,
        'time_offset': time_offset,
        'root_node': None,
        'optimistic_cost_table': None,
        'terminal_node': None,
        'PEs': PES
    }
    _self = SimpleNamespace(**_self)
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

    for task in ready_queue:
        if task.jobID not in common.deadline_dict:
            common.deadline_dict[task.jobID] = common.arrive_time[task.jobID] + deadline
    for proc in proc_schedules:
        _self.numExistingJobs = _self.numExistingJobs + len(proc_schedules[proc])

    # 不进入该分支
    if relabel_nodes:
        dag = nx.relabel_nodes(dag, dict(map(lambda node: (node, node+_self.numExistingJobs), list(dag.nodes()))))
    # 进入该分支
    else:
        #Negates any offsets that would have been needed had the jobs been relabeled
        _self.numExistingJobs = 0

    for i in range(_self.numExistingJobs + len(_self.computation_matrix)):
        _self.task_schedules[i] = None
    for i in range(len(_self.communication_matrix)):
        if i not in _self.proc_schedules:
            _self.proc_schedules[i] = []

    for proc in proc_schedules:
        for schedule_event in proc_schedules[proc]:
            _self.task_schedules[schedule_event.task] = schedule_event

    # Nodes with no successors cause the any expression to be empty    
    root_node = [node for node in dag.nodes() if not any(True for _ in dag.predecessors(node))]
    root_node = root_node[0]
    _self.root_node = root_node

    start_time = int(round(time.time() * 1000))
    # 计算得到OCT和各节点的rankoct
    _self.optimistic_cost_table = _compute_optimistic_cost_table(_self, dag)

    # 按非递增序列以rankoct为节点排序
    sorted_nodes = sorted(dag.nodes(), key=lambda node: dag.nodes()[node]['rank'], reverse=True)
    # 根节点必须是第一个
    if sorted_nodes[0] != root_node:
        # idx为root_node的index
        idx = sorted_nodes.index(root_node)
        # Cyclically rotate the sorted nodes between sorted_nodes[0] and sorted_nodes[idx]
        # This ensures that relative ordering between nodes of equivalent rank (i.e. a child of a cost-zero parent) are preserved
        # Namely, if all of these nodes are in front of the root node, then (I think) they must _all_ have the same cost and can thus be rotated freely
        if idx > 1:
            # 将root_node插入第一位,别的按序后移一位
            sorted_nodes[0:idx+1] = [sorted_nodes[idx]] + sorted_nodes[0:idx]
        else:
            # idx = 1
            sorted_nodes[idx], sorted_nodes[0] = sorted_nodes[0], sorted_nodes[idx]
    for node in sorted_nodes:
        if _self.task_schedules[node] is not None:
            continue
        #task start end proc
        minTaskSchedule = ScheduleEvent(node, inf, inf, -1)
        minOptimisticCost = inf
        for proc in range(len(communication_matrix)):
            # 得到每个node在各proc中执行的情况
            taskschedule = _compute_eft(_self, dag, node, proc)
            # Oeft(node,processor) = EFT(node,processor) + OCT(node,processor),得到每个节点最小的Oeft(node,processor)
            if (taskschedule.end + _self.optimistic_cost_table[node][proc] < minTaskSchedule.end + minOptimisticCost):
                minTaskSchedule = taskschedule
                minOptimisticCost = _self.optimistic_cost_table[node][proc]
        _self.task_schedules[node] = minTaskSchedule
        # processor执行队列添加新的调度工作
        _self.proc_schedules[minTaskSchedule.proc].append(minTaskSchedule)
        # 按结束时间有效到大排序执行任务
        _self.proc_schedules[minTaskSchedule.proc] = sorted(_self.proc_schedules[minTaskSchedule.proc], key=lambda schedule_event: schedule_event.end)
    end_time = int(round(time.time() * 1000))
    common.exe_time += end_time - start_time
    dict_output = {}
    # proc_num:processor的序号
    # proc_tasks:processor中要执行的任务序列
    flag=False
    for proc_num, proc_tasks in _self.proc_schedules.items():
        # idx:对应processor要执行任务序列的每一个index
        # task:对应每个任务
        for idx, task in enumerate(proc_tasks):
            if task.task>=26:
                flag=True
            # 若该任务非processor中task执行队列的首个任务,则传入[前一个任务的id]
            if idx > 0 and (proc_tasks[idx-1].end - proc_tasks[idx-1].start > 0):
                dict_output[task.task] = (proc_num, idx, [proc_tasks[idx-1].task])
            # 若该任务为processor中task执行队列的首个任务,则执行队列传入[]
            else:
                dict_output[task.task] = (proc_num, idx, [])

    if len(common.output)==0:
        common.output.update(_self.proc_schedules)
    else:
        for i in range(len(_self.proc_schedules)):
            for task in common.output[i]:
                for t in _self.proc_schedules[i]:
                    if task.task==t.task:
                        if(task.end <= t.end):
                            task._replace(end = t.end)
                            _self.proc_schedules[i].remove(t)


            common.output[i].extend(_self.proc_schedules[i])
    return _self.proc_schedules, _self.task_schedules, dict_output

def _compute_optimistic_cost_table(_self, dag):
    """
    Uses a basic BFS approach to traverse upwards through the graph building the optimistic cost table along the way
    """

    optimistic_cost_table = {}

    terminal_node = [node for node in dag.nodes() if not any(True for _ in dag.successors(node))]
    terminal_node = terminal_node[0]
    _self.terminal_node = terminal_node

    diagonal_mask = np.ones(_self.communication_matrix.shape, dtype=bool)
    np.fill_diagonal(diagonal_mask, 0)
    # DAG的平均bandwidth
    avgCommunicationCost = np.mean(_self.communication_matrix[diagonal_mask])
    for edge in dag.edges():
        nx.set_edge_attributes(dag, { edge: float(dag.get_edge_data(*edge)['weight']) / avgCommunicationCost }, 'avgweight')

    # OCT(terminal_node,Pi) = 0
    optimistic_cost_table[terminal_node] = _self.computation_matrix.shape[1] * [0]
    # lol whoops dag.node doesn't exist
    #dag.node[terminal_node]['rank'] = 0

    # rankoct(terminal_node) = 0
    nx.set_node_attributes(dag, { terminal_node: 0 }, "rank")
    visit_queue = deque(dag.predecessors(terminal_node))

    # all()中元素除了是 0、空、None、False 外都算 True
    # 判断node节点的后继节点oct是否均已计算
    node_can_be_processed = lambda node: all(successor in optimistic_cost_table for successor in dag.successors(node))
    while visit_queue:
        node = visit_queue.pop()

        # 找到后续节点的oct均已计算的队列中的node
        while node_can_be_processed(node) is not True:
            try:
                node2 = visit_queue.pop()
            except IndexError:
                raise RuntimeError(f"Node {node} cannot be processed, and there are no other nodes in the queue to process instead!")
            visit_queue.appendleft(node)
            node = node2

        # 此时node为可计算oct的节点
        optimistic_cost_table[node] = _self.computation_matrix.shape[1] * [0]


        # Perform OCT kernel
        # Need to build the OCT entries for every task on each processor

        # 第一层循环计算各PE的OCT
        for curr_proc in range(_self.computation_matrix.shape[1]):
            # Need to maximize over all the successor nodes
            max_successor_oct = -inf
            # 遍历各后继节点，计算目标最大值
            for succnode in dag.successors(node):
                # Need to minimize over the costs across each processor
                min_proc_oct = inf
                # 便利各processor，计算目标最小值
                for succ_proc in range(_self.computation_matrix.shape[1]):
                    # OCT(succnode,succ_proc)
                    successor_oct = optimistic_cost_table[succnode][succ_proc]
                    # w(succnode,succ_proc)
                    successor_comp_cost = _self.computation_matrix[succnode][succ_proc]
                    # avg(c(node,succnode))，当前processor为后继processor时为0
                    successor_comm_cost = dag[node][succnode]['avgweight'] if curr_proc != succ_proc else 0
                    # 临时值
                    cost = successor_oct + successor_comp_cost + successor_comm_cost
                    # 取最小的cost
                    if cost < min_proc_oct:
                        min_proc_oct = cost
                # 最内层循环结束，得到最小值
                # 取后继节点计算成的最大的最小值作为oct
                if min_proc_oct > max_successor_oct:
                    max_successor_oct = min_proc_oct
            optimistic_cost_table[node][curr_proc] = max_successor_oct
        # End OCT kernel
        # lol whoops dag.node doesn't exist
        #dag.node[node]['rank'] = np.mean(optimistic_cost_table[node])
        # 普通节点的rankoct为各processor下oct的平均值
        nx.set_node_attributes(dag, { node: np.mean(optimistic_cost_table[node]) }, "rank")
        # 更新队列，将计算好的节点的前驱节点放入队列中，且不要重复放置
        visit_queue.extendleft([prednode for prednode in dag.predecessors(node) if prednode not in visit_queue])

    return optimistic_cost_table

def _compute_eft(_self, dag, node, proc):
    """
    Computes the EFT of a particular node if it were scheduled on a particular processor
    It does this by first looking at all predecessor tasks of a particular node and determining the earliest time a task would be ready for execution (ready_time)
    It then looks at the list of tasks scheduled on this particular processor and determines the earliest time (after ready_time) a given node can be inserted into this processor's queue
    """
    ready_time = _self.time_offset
    # 便利node的前驱节点
    for prednode in list(dag.predecessors(node)):
        # preadjob为prenode的调度工作
        predjob = _self.task_schedules[prednode]
        # ready_time_t=AFT(prednode)+Cnode,prednode
        if _self.communication_matrix[predjob.proc, proc] == 0:
            ready_time_t = predjob.end
        else:
            ready_time_t = predjob.end + dag[predjob.task][node]['weight'] / _self.communication_matrix[predjob.proc, proc]
        # 取max{AFT(prednode)+Cnode,prednode}
        if ready_time_t > ready_time:
            ready_time = ready_time_t

    # 当前processor对node执行时间
    computation_time = _self.computation_matrix[node-_self.numExistingJobs, proc]
    # 当前processor下的task执行队列
    job_list = _self.proc_schedules[proc]
    for idx in range(len(job_list)):
        prev_job = job_list[idx]
        if idx == 0:
            # 可插入该processor中task执行队列首个任务之前
            if (prev_job.start - computation_time) - ready_time > 0:
                job_start = ready_time
                min_schedule = ScheduleEvent(node, job_start, job_start+computation_time, proc)
                break
        # 如果遍历到队尾则直接可判别为不可插入,只能在队尾之后加上
        if idx == len(job_list)-1:
            # 不可插入执行队列中任何task之前时,prev_job.end为avail[proc]
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
        # 除队首队尾以外,每次判断当前节点之后能否插入
        next_job = job_list[idx+1]
        # max(ready_time, prev_job.end) = EST, 如果该节点EST与执行时间之和要早于下一个节点开始时间,则可插入
        if (next_job.start - computation_time) - max(ready_time, prev_job.end) >= 0:
            job_start = max(ready_time, prev_job.end)
            min_schedule = ScheduleEvent(node, job_start, job_start + computation_time, proc)
            break
    # 若当前processor中task执行队列为空,直接判断
    else:
        #For-else loop: the else executes if the for loop exits without break-ing, which in this case means the number of jobs on this processor are 0
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
        contentsList = contentsList[0:len(contentsList)-1] if contentsList[len(contentsList)-1] == [''] else contentsList

        matrix = np.array(contentsList)
        matrix = np.delete(matrix, 0, 0) # delete the first row (entry 0 along axis 0)
        matrix = np.delete(matrix, 0, 1) # delete the first column (entry 0 along axis 1)
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
    parser = argparse.ArgumentParser(description="A tool for finding PEFT schedules for given DAG task graphs")
    parser.add_argument("-d", "--dag_file",
                        help="File containing input DAG to be scheduled. Uses default 10 node dag from Arabnejad 2014 if none given.",
                        type=str, default="test/peftgraph_task_connectivity.csv")
    parser.add_argument("-p", "--pe_connectivity_file",
                        help="File containing connectivity/bandwidth information about PEs. Uses a default 3x3 matrix from Arabnejad 2014 if none given.",
                        type=str, default="test/peftgraph_resource_BW.csv")
    parser.add_argument("-t", "--task_execution_file",
                        help="File containing execution times of each task on each particular PE. Uses a default 10x3 matrix from Arabnejad 2014 if none given.",
                        type=str, default="test/peftgraph_task_exe_time.csv")
    parser.add_argument("-l", "--loglevel",
                        help="The log level to be used in this module. Default: INFO",
                        type=str, default="INFO", dest="loglevel", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--showDAG",
                        help="Switch used to enable display of the incoming task DAG",
                        dest="showDAG", action="store_true")
    parser.add_argument("--showGantt",
                        help="Switch used to enable display of the final scheduled Gantt chart",
                        dest="showGantt", action="store_true")
    return parser
def calculateCost(_self, dag):
    processors = list(_self.proc_schedules.keys())
    WEC = 0
    for idx, proc in enumerate(processors):
        if len(_self.proc_schedules[proc]) == 0:
            continue
        t0 = _self.proc_schedules[proc][0]
        tn = _self.proc_schedules[proc][-1]
        LST = 0
        LFT = 0
        if t0 == _self.root_node:
            LST = t0.start
        else:
            TT = 0
            for pred in dag.predecessors(t0.task):
                if dag[pred][t0.task]['avgweight'] > TT:
                    TT = dag[pred][t0.task]['avgweight']
            LST = t0.start - TT

        if tn == _self.terminal_node:
            LFT = tn.end
        else:
            TT = 0
            for succ in dag.successors(tn.task):
                if dag[tn.task][succ]['avgweight'] > TT:
                    TT = dag[tn.task][succ]['avgweight']
            LFT = tn.end + TT

        WEC += (LFT - LST) / 5 * float(_self.PEs[proc].cost)
    common.cost += WEC
    common.nums += 1
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

    processor_schedules, _, _ = schedule_dag(dag, communication_matrix=communication_matrix, computation_matrix=computation_matrix)
    for proc, jobs in processor_schedules.items():
        logger.info(f"Processor {proc} has the following jobs:")
        logger.info(f"\t{jobs}")
    # if args.showGantt:
    showGanttChart(processor_schedules)
