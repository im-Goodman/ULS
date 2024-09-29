import matplotlib.pyplot as plt
import csv
from docplex.cp.model import *
import docplex.cp.utils_visu as visu
from docplex.cp.config import context
import docplex.cp.parameters as params
import common

def CP_Multi(env_time, P_elems, resource_matrix, domain_applications, generated_jobs):
    ###### Step 1 - Initialize variable and parameters
    plt.close('all')
    
    # Set docplex related parameters
    context.solver.trace_log = False
    #params.CpoParameters.OptimalityTolerance = 2
    #params.CpoParameters.RelativeOptimalityTolerance= 2
    
    Dags = []
    Dags_2 = {}
    # Get the task in Outstanding and Ready Queues
    # Since tasks in Completed Queue are already done, they will not be considered
    for task in common.TaskQueues.outstanding.list:
        if task.jobID not in Dags:
            Dags.append(task.jobID)
            for i in range(len(domain_applications.list)):
                name = domain_applications.list[i].name
                if name == task.jobname:
                    Dags_2[task.jobID] = {}
                    Dags_2[task.jobID]['selection'] = i
    for task in common.TaskQueues.ready.list:
        if task.jobID not in Dags:
            Dags.append(task.jobID)
            for i in range(len(domain_applications.list)):
                name = domain_applications.list[i].name
                if name == task.jobname:
                    Dags_2[task.jobID] = {}
                    Dags_2[task.jobID]['selection'] = i
    #Dags.sort()

    common.ilp_job_list = [(key,Dags_2[key]['selection']) for key in Dags_2.keys()]
    print(Dags_2)
    
    
    NbDags = len(Dags)                                                      # Current number of jobs in the system
    PEs = []                                                                # List of PE element in the given SoC configuration                                                

    print('[I] Time %d: There is %d job(s) in the system' %(env_time, NbDags)   )
    print('[D] Time %d: ID of the jobs in the system are' %(env_time),Dags)
   
    ###### Step 2 - Prepare Data 
    
    # First, check if there are any tasks currently being executed on a PE
    # if yes, retrieve remaining time of execution on that PE and
    # do not assign a task during ILP solution
    for i, PE in enumerate(P_elems):
        if (PE.type == 'MEM') or (PE.type == 'CAC') :                       # Do not consider Memory ond Cache
            continue
        PEs.append(PE.name)                                                 # Populate the name of the PEs in the SoC
    #print(PEs)
    
    for ID in Dags_2.keys():
        Dags_2[ID]['Tasks'] = []                                            # list of tasks that CPLEX will return a schedule for
        Dags_2[ID]['Functionality'] = []                                    # list of task-PE relationship
        Dags_2[ID]['Con_Precedence'] = []                                    # list of task dependencies

        app_id = Dags_2[ID]['selection']        
        #print(len(Dags), len(self.generated_job_list))
        num_of_tasks = len(domain_applications.list[app_id].task_list)
        
        for task in domain_applications.list[app_id].task_list:
            Dags_2[ID]['Tasks'].append(task.base_ID)
            
            # Next, gather the information about which PE can run which Tasks 
            # and if so, the expected execution time
            for resource in resource_matrix.list:
                if (resource.type == 'MEM') or (resource.type == 'CAC') :        # Do not consider Memory ond Cache
                    continue
                else:
                    if task.name in resource.supported_functionalities:
                        ind = resource.supported_functionalities.index(task.name)
                        #Functionality.append((resource.name, task.base_ID, resource.performance[ind]))
                        Dags_2[ID]['Functionality'].append((resource.name, task.base_ID, resource.performance[ind]))
            
            # Finally, gather dependency information between tasks
            
            for i,predecessor in enumerate(task.predecessors):
                #print(task.ID, predecessor, num_of_tasks, last_ID, task.base_ID)

                for resource in resource_matrix.list:
                    if (resource.type == 'MEM') or (resource.type == 'CAC') :
                        continue
                    else:
                        #pred_name = self.generated_job_list[job_ind].task_list[predecessor-last_ID].name
                        pred_name = domain_applications.list[app_id].task_list[ predecessor- (task.ID - task.base_ID) ].name
                        if (pred_name in resource.supported_functionalities):
                            c_vol = domain_applications.list[app_id].comm_vol[predecessor - (task.ID - task.base_ID), task.base_ID]
                            Dags_2[ID]['Con_Precedence'].append((resource.name,Dags_2[ID]['Tasks'][predecessor - (task.ID - task.base_ID)], task.base_ID, c_vol))

    #print(Dags_2[ID]['Tasks'])
    #print(len(Dags_2[ID]['Functionality']))
    #print(Dags_2[ID]['Con_Precedence'])
    

    
    ###### Step 3 - Create the model
    mdl = CpoModel()
    
    # Create dag interval variables
    dags = { d : mdl.interval_var(name="dag"+str(d)) for d in Dags_2.keys()}
    #print(dags)
    
    # Create tasks interval variables and pe_tasks interval variables
    # pe_tasks are optional and only one of them will be mapped to the corresponding
    # tasks interval variable
    # For example, task_1 has 3 pe_tasks (i.e.,P1-task_1, P2-task_1, P3, task_1)
    # only of these will be selected. If the first one is selected, it means that task_1 will be executed in P1
    tasks_2 = {}
    pe_tasks_2 = {}
    task_names_ids_2 = {}
    last_ID = 0
    for d in Dags_2.keys():
        num_of_tasks = len(generated_jobs[d].task_list)
        for t in Dags_2[d]['Tasks']:
            tasks_2[(d,t)] = mdl.interval_var(name = str(d)+"_"+str(t))
        for f in Dags_2[d]['Functionality']:
            #print(f)
            name = str(d)+"_"+str(f[1])+'-'+str(f[0])
            if len(common.TaskQueues.running.list) == 0 and len(common.TaskQueues.completed.list) == 0:
                pe_tasks_2[(d,f)] = mdl.interval_var(optional=True, size =int(f[2]), name = name )
                task_names_ids_2[name] = last_ID+f[1]
            else:
                for ii, running_task in enumerate(common.TaskQueues.running.list):
                    if (d == running_task.jobID) and (f[1] == running_task.base_ID) and (f[0] == P_elems[running_task.PE_ID].name):

                        ind = resource_matrix.list[running_task.PE_ID].supported_functionalities.index(running_task.name)
                        exec_time = resource_matrix.list[running_task.PE_ID].performance[ind]
                        free_time = int(running_task.start_time + exec_time - env_time)
                        #print(free_time)    
                        pe_tasks_2[(d,f)] = mdl.interval_var(optional=True, start=0, end=free_time,name = name )
                        task_names_ids_2[name] = last_ID+f[1]
                        break
                    elif (d == running_task.jobID) and (f[1] == running_task.base_ID):
                        pe_tasks_2[(d,f)] = mdl.interval_var(optional=True, size =INTERVAL_MAX, name = name )
                        task_names_ids_2[name] = last_ID+f[1]
                        break
                else:
                    pe_tasks_2[(d,f)] = mdl.interval_var(optional=True, size =int(f[2]), name = name )
                    task_names_ids_2[name] = last_ID+f[1]
                
                for iii, completed_task in enumerate(common.TaskQueues.completed.list):
                    if (d == completed_task.jobID) and (f[1] == completed_task.base_ID) and (f[0] == P_elems[completed_task.PE_ID].name):
                        #print(completed_task.name)
                        #pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =0, name = name )
                        pe_tasks_2[(d,f)] = mdl.interval_var(optional=True, start=0, end=0, name = name )
                        task_names_ids_2[name] = last_ID+f[1]
                    elif (d == completed_task.jobID) and (f[1] == completed_task.base_ID):
                        pe_tasks_2[(d,f)] = mdl.interval_var(optional=True, size =INTERVAL_MAX, name = name )
                        task_names_ids_2[name] = last_ID+f[1]
                #print('3',pe_tasks[(d,f)])
                  
        last_ID += num_of_tasks
    #print(tasks_2)
    #print(task_names_ids_2)
    
    # Add the temporal constraints            
    for d in Dags_2.keys():
        for c in Dags_2[d]['Con_Precedence']:
            for (p1, task1, d1) in Dags_2[d]['Functionality']:
                if p1 == c[0] and task1 == c[1]:
                    p1_id = PEs.index(p1)
                    for (p2, task2, d2) in Dags_2[d]['Functionality']:
                        if p2 == c[0] and task2 == c[2]:
                            mdl.add( mdl.end_before_start(pe_tasks_2[d,(p1,task1,d1)], pe_tasks_2[d,(p2,task2,d2)], 0) )
                        elif p2 != c[0] and task2 == c[2]:
                            p2_id = PEs.index(p2)
                            #print(p2_id)
                            bandwidth = common.ResourceManager.comm_band[p1_id,p2_id]
                            comm_time = int( (c[3])/bandwidth )
                            for ii, completed_task in enumerate(common.TaskQueues.completed.list):
                                if ((d == completed_task.jobID) and (task1 == completed_task.base_ID) and (p1 == P_elems[completed_task.PE_ID].name)):
                                    mdl.add( mdl.end_before_start(pe_tasks_2[d,(p1,task1,d1)], pe_tasks_2[d,(p2,task2,d2)], max(0,comm_time+completed_task.finish_time-env_time)  ))
                                    #print (d, p1,p2, task1,task2, max(0,int(c[3])+completed_task.finish_time-self.env.now) )
                                    break
                            else:
                                #print(d, p1,p2, task1,task2, c[3])
                                mdl.add( mdl.end_before_start(pe_tasks_2[d,(p1,task1,d1)], pe_tasks_2[d,(p2,task2,d2)], comm_time ) )
                

    # Add the span constraints
    # This constraint enables to identify tasks in a dag            
    for d in Dags_2.keys():
        mdl.add( mdl.span(dags[d], [tasks_2[(d,t)] for t in Dags_2[d]['Tasks'] ] ) )
        
    
    # Add the alternative constraints
    # This constraint ensures that only one PE is chosen to execute a particular task                
    for d in Dags_2.keys():
        for t in Dags_2[d]['Tasks']:
            mdl.add( mdl.alternative(tasks_2[d,t], [pe_tasks_2[d,f] for f in Dags_2[d]['Functionality'] if f[1]==t]) )
            
    
    # Add the no overlap constraints
    # This constraint ensures that there will be no overlap for the task being executed on the same PE        
    for p in PEs:
        b_list = [pe_tasks_2[d,f] for d in Dags_2.keys() for f in Dags_2[d]['Functionality'] if f[0]==p]
        if b_list:
            mdl.add( mdl.no_overlap([pe_tasks_2[d,f] for d in Dags_2.keys() for f in Dags_2[d]['Functionality'] if f[0]==p]))
        else:
            continue     
    
    # Add the objective
    mdl.add(mdl.minimize(mdl.sum([mdl.end_of(dags[d]) for i,d in enumerate(Dags)])))
    #mdl.add(mdl.minimize(mdl.max([mdl.end_of(pe_tasks_2[(d,f)]) for i,d in enumerate(Dags) for f in Dags_2[d]['Functionality'] ])))
    #mdl.add(mdl.minimize(mdl.max(mdl.end_of(dags[d]) for i,d in enumerate(Dags))))
    
    ###### Step 4 - Solve the model and print some results
    # Solve the model
    print("\nSolving CP model....")
    msol = mdl.solve(TimeLimit = 600)
    #msol = mdl.solve()
    #print("Completed")

    #print(msol.print_solution())
    #print(msol.is_solution_optimal())
    print(msol.get_objective_gaps())
    print(msol.get_objective_values()[0])
    
    
    tem_list = []
    for d in Dags:
        #for f in Functionality:
        for f in Dags_2[d]['Functionality']:
            #solution = msol.get_var_solution(pe_tasks[(d,f)])
            solution = msol.get_var_solution(pe_tasks_2[(d,f)])
            if solution.is_present():
                #ID = task_names_ids[solution.get_name()]
                ID = task_names_ids_2[solution.get_name()]
                tem_list.append( (ID, f[0], solution.get_start(), solution.get_end()) )
    tem_list.sort(key=lambda x: x[2], reverse=False)
    #print(tem_list)
    
    actual_schedule = []
    for i,p in enumerate(PEs):
        count = 0      
        for item in tem_list:
            if item[1] == p:
                actual_schedule.append( (item[0],i,count+1))
                count += 1
    actual_schedule.sort(key=lambda x: x[0], reverse=False)
    #print(actual_schedule)
    
    common.table = []
    for element in actual_schedule:
        common.table.append((element[1],element[2]))
    #print(common.table)    
    #print(len(common.table))
    
    
    if (common.simulation_mode == 'validation'):
        colors = ['salmon','turquoise', 'lime' , 'coral', 'lightpink']
        PEs.reverse()                     
        for i,p in enumerate(PEs):
            #visu.panel()
            #visu.pause(PE_busy_times[p])
            visu.sequence(name=p)

            for ii,d in enumerate(Dags):
                #for f in Functionality: 
                for f in Dags_2[d]['Functionality']:
                    #wt = msol.get_var_solution(pe_tasks[(d,f)])
                    wt = msol.get_var_solution(pe_tasks_2[(d,f)])
                    if wt.is_present() and p == f[0]:
                        color = colors[ii%len(colors)]
                        #visu.interval(wt, color, str(task_names_ids[wt.get_name()]))
                        visu.interval(wt, color, str(task_names_ids_2[wt.get_name()]))          
        visu.show()
    

    for d in Dags:
        for task in generated_jobs[d].task_list:

            task_sched_ID = 0
            task.dynamic_dependencies.clear()                                  # Clear dependencies from previosu ILP run
           
            ind = Dags.index(d)
            for i in range(ind):
                selection = Dags_2[Dags[i]]['selection']
                task_sched_ID += len(domain_applications.list[selection].task_list)
            task_sched_ID += task.base_ID
            
            task_order = common.table[task_sched_ID][1]
           
            for k in Dags:

                for dyn_depend in generated_jobs[k].task_list:
                    dyn_depend_sched_ID = 0
                   
                    ind_k = Dags.index(k)
                    for ii in range(ind_k):
                        selection = Dags_2[Dags[ii]]['selection']
                        dyn_depend_sched_ID += len(domain_applications.list[selection].task_list)
                    dyn_depend_sched_ID += dyn_depend.base_ID    
                       
                    if ( (common.table[dyn_depend_sched_ID][0] == common.table[task_sched_ID][0]) and 
                        (common.table[dyn_depend_sched_ID][1] == task_order-1) and 
                        (dyn_depend.ID not in task.predecessors) and 
                        (dyn_depend.ID not in task.dynamic_dependencies) ):
                        
                        task.dynamic_dependencies.append(dyn_depend.ID)
                #print(task.ID, task.dynamic_dependencies)

# end of CP_Multi(......

###########################################################################################################################
###########################################################################################################################


def CP_Cluster(env_time, P_elems, resource_matrix, domain_applications, generated_jobs):
    ###### Step 1 - Initialize variable and parameters
    plt.close('all')
    num_of_tasks = len(generated_jobs[-1].task_list)
    
    # Set docplex related parameters
    context.solver.trace_log = False
    #params.CpoParameters.OptimalityTolerance = 2
    #params.CpoParameters.RelativeOptimalityTolerance= 2
    
    Dags = []
    # Get the task in Outstanding and Ready Queues
    # Since tasks in Completed Queue are already done, they will not be considered
    for task in common.TaskQueues.outstanding.list:
        if task.jobID not in Dags:
            Dags.append(task.jobID)
    for task in common.TaskQueues.ready.list:
        if task.jobID not in Dags:
            Dags.append(task.jobID)
    Dags.sort()
    #print(Dags)
    common.ilp_job_list = Dags

    
    
    NbDags = len(Dags)                                                          # Current number of jobs in the system
    Tasks = []                                                                  # list of tasks that CPLEX will return a schedule for
    Cluster = []                                                                # list of PE clusters in the given SoC configuration
    Capacity = []                                                               # list of parallel processing capacity of each cluster
    PE_IDs = []                                                                 # list of IDs of PEs in a cluster
    Func_cluster = []                                                           # list of task-PE relationship
    Precedence_cluster = []                                                     # list of task dependencies
    
    
    print('[I] Time %d: There is %d job(s) in the system' %(env_time, NbDags))
    print('[D] Time %d: ID of the jobs in the system are' %(env_time),Dags)
    
    ###### Step 2 - Prepare Data 
    
    # First, check if there are any tasks currently being executed on a PE
    # if yes, retrieve remaining time of execution on that PE and
    # do not assign a task during ILP solution
    for i, PE in enumerate(P_elems):
        if (PE.type == 'MEM') or (PE.type == 'CAC') :                       # Do not consider Memory ond Cache
            continue
        else: 
            if not PE.type in Cluster:
                Cluster.append(PE.type)
                Capacity.append(1)
                PE_IDs.append([PE.ID])
            else:
                ind = Cluster.index(PE.type)
                Capacity[ind] += 1
                PE_IDs[ind].append(PE.ID)
 
    #print(Capacity)
    #print(Cluster)
    #print(PE_IDs)
    
    # This only supports streaming jobs from one application
    for task in generated_jobs[-1].task_list:
        if task.base_ID not in Tasks:
            Tasks.append(task.base_ID)                                      # Populate the base ID of the task 
            
        # Next, gather the information about which PE can run which Tasks 
        # and if so, the expected execution time
        for resource in resource_matrix.list:
            if (resource.type == 'MEM') or (resource.type == 'CAC'):        # Do not consider Memory ond Cache
                continue
            else:           
                if task.name in resource.supported_functionalities:
                    ind = resource.supported_functionalities.index(task.name)                  
                    data = (resource.type, task.base_ID, resource.performance[ind])
                    if data in Func_cluster:
                        continue
                    else:
                        Func_cluster.append(data)                           # Populate PE-task pairs (at cluster level)

        # Finally, gather dependency information between tasks
        for i,predecessor in enumerate(task.predecessors):
            #print(task.ID, predecessor)
            for resource in resource_matrix.list:
                if (resource.type == 'MEM') or (resource.type == 'CAC') :
                    continue
                else:
                    pred_name = generated_jobs[-1].task_list[predecessor%num_of_tasks].name
                    #print(task.name, pred_name)
                    if (pred_name in resource.supported_functionalities):
                        #print(resource.name,Tasks[predecessor], task.name)
                        c_vol = generated_jobs[0].comm_vol[predecessor%num_of_tasks, task.base_ID]
                        #print(resource.name,Tasks[predecessor], task.name, c_vol)
                     
                        data = (resource.type,Tasks[predecessor%num_of_tasks], task.base_ID, c_vol)
                        if data in Precedence_cluster:
                            continue
                        else:
                            Precedence_cluster.append(data)                 # Populate task dependency relation
    # end of for task in generated_jobs....
                                          
    #print(Tasks)
    #print(Func_cluster)
    #print(Precedence_cluster)
    
    ###### Step 3 - Create the model
    mdl = CpoModel()
    
    # Create dag interval variables
    dags = { d : mdl.interval_var(name="dag"+str(d)) for d in Dags}
    #print(dags)
    
    # Create tasks interval variables and pe_tasks interval variables
    # pe_tasks are optional and only one of them will be mapped to the corresponding
    # tasks interval variable
    # For example, task_1 has 3 pe_tasks (i.e.,P1-task_1, P2-task_1, P3, task_1)
    # only of these will be selected. If the first one is selected, it means that task_1 will be executed in P1
    tasks = {}
    pe_tasks = {}
    task_names_ids = {}
    for d in Dags:
        for i,t in enumerate(Tasks):
            tasks[(d,t)] = mdl.interval_var(name = str(d)+"_"+str(t)) 
        for f in Func_cluster:
            #print(f[0])
            ind = Cluster.index(f[0])
            PE_list = PE_IDs[ind]
            #print(PE_list)
            name = str(d)+"_"+str(f[1])+'-'+str(f[0])
            
            if len(common.TaskQueues.running.list) == 0 and len(common.TaskQueues.completed.list) == 0:
                pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =int(f[2]), name = name)
                task_names_ids[name] = d*num_of_tasks+f[1]

            else:
                for ii, running_task in enumerate(common.TaskQueues.running.list):
                    #if (d == running_task.jobID) and (f[1] == running_task.base_ID) and (f[0] == P_elems[running_task.PE_ID].name):
                    if (d == running_task.jobID) and (f[1] == running_task.base_ID) and (running_task.PE_ID in PE_list):
                        
                        ind = resource_matrix.list[running_task.PE_ID].supported_functionalities.index(running_task.name)
                        exec_time = resource_matrix.list[running_task.PE_ID].performance[ind]
                        free_time = int(running_task.start_time + exec_time - env_time)
                        #print(running_task.start_time, exec_time, env_time, free_time)

                        pe_tasks[(d,f)] = mdl.interval_var(optional=True, start=0, end=free_time,name = name )
                        task_names_ids[name] = d*num_of_tasks+f[1]
                        break
                    elif (d == running_task.jobID) and (f[1] == running_task.base_ID):
                        pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =INTERVAL_MAX, name = name )
                        task_names_ids[name] = d*num_of_tasks+f[1]
                        break
                else:
                    pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =int(f[2]), name = name )
                    task_names_ids[name] = d*num_of_tasks+f[1]
                
                for iii, completed_task in enumerate(common.TaskQueues.completed.list):
                    #if (d == completed_task.jobID) and (f[1] == completed_task.base_ID) and (f[0] == P_elems[completed_task.PE_ID].name):
                    if (d == completed_task.jobID) and (f[1] == completed_task.base_ID) and (completed_task.PE_ID in PE_list): 
                        #print(completed_task.name)
                        pe_tasks[(d,f)] = mdl.interval_var(optional=True, start=0, end=0, name = name )
                        task_names_ids[name] = d*num_of_tasks+f[1]
                    elif (d == completed_task.jobID) and (f[1] == completed_task.base_ID):
                        pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =INTERVAL_MAX, name = name )
                        task_names_ids[name] = d*num_of_tasks+f[1]
           
    #print(tasks)                    
    #print((pe_tasks))
    #print(task_names_ids)
    
    
    # Add the temporal constraints    
    for d in Dags:
        #for c in Con_Precedence:
        for c in Precedence_cluster:
            #print (c)
            for (p1, task1, d1) in Func_cluster:
                if p1 == c[0] and task1 == c[1]:
                    ind_1 = Cluster.index(p1)
                    PE_list_1 = PE_IDs[ind_1]
                    for (p2, task2, d2) in Func_cluster:
                        if p2 == c[0] and task2 == c[2]:
                            #print (d,p1,p2, task1,task2, 0)
                            #print(pe_tasks[d,(p1,task1,d1)])
                            #print(pe_tasks[d,(p2,task2,d2)])
                            mdl.add( mdl.end_before_start(pe_tasks[d,(p1,task1,d1)], pe_tasks[d,(p2,task2,d2)], 0) )
                        elif p2 != c[0] and task2 == c[2]:
                            ind_2 = Cluster.index(p2)
                            PE_list_2 = PE_IDs[ind_2]
                            bandwidth = common.ResourceManager.comm_band[PE_list_1[-1],PE_list_2[-1]]
                            comm_time = int( (c[3])/bandwidth )
                            
                            for ii, completed_task in enumerate(common.TaskQueues.completed.list):
                                
                                #if ((d == completed_task.jobID) and (task1 == completed_task.base_ID) and (p1 == P_elems[completed_task.PE_ID].name)):
                                if ((d == completed_task.jobID) and (task1 == completed_task.base_ID) and (completed_task.PE_ID in PE_list_1)):
                                    mdl.add( mdl.end_before_start(pe_tasks[d,(p1,task1,d1)], pe_tasks[d,(p2,task2,d2)], max(0,comm_time+completed_task.finish_time-env_time)  ))
                                    #print (d, p1,p2, task1,task2, max(0,int(c[3])+completed_task.finish_time-env_time) )
                                    break
                            else:
                                #print(d, p1,p2, task1,task2, c[3])
                                mdl.add( mdl.end_before_start(pe_tasks[d,(p1,task1,d1)], pe_tasks[d,(p2,task2,d2)], comm_time ) )
    
    # Add end of dags constraints
    for i,d in enumerate(Dags):
        for j in range(i):
            #print(j,i)
            mdl.add ( mdl.end_of(dags[Dags[j]]) <= mdl.end_of(dags[Dags[i]]) )
            #mdl.add ( mdl.start_of(dags[Dags[j]]) <= mdl.start_of(dags[Dags[i]]) )
    
    # Add the span constraints
    # This constraint enables to identify tasks in a dag
    for d in Dags:
        mdl.add( mdl.span(dags[d], [tasks[(d,t)] for t in Tasks ] ) )
    

    # Add the alternative constraints
    # This constraint ensures that only one PE is chosen to execute a particular task
    for d in Dags:
        for t in Tasks:
            mdl.add( mdl.alternative(tasks[d,t], [pe_tasks[d,f] for f in Func_cluster if f[1]==t]) )    
    
    # Constrain capacity of resources
    for r in range(len(Cluster)):
        cluster = [mdl.pulse(pe_tasks[(d,f)], 1) for d in Dags for f in Func_cluster if Cluster[r] == f[0] ]
        mdl.add(mdl.sum(cluster) <= Capacity[r])
   

    # Add the objective
    #mdl.add(mdl.minimize(mdl.sum([mdl.end_of(dags[d])*(i+1) for i,d in enumerate(Dags)])))
    #mdl.add(mdl.minimize(mdl.sum([mdl.end_of(dags[d])*(len(Dags)-i) for i,d in enumerate(Dags)])))
    mdl.add(mdl.minimize(mdl.sum([mdl.end_of(dags[d]) for i,d in enumerate(Dags)])))
    #mdl.add(mdl.minimize(mdl.max([mdl.end_of(pe_tasks[(d,f)]) for i,d in enumerate(Dags) for f in Functionality])))
    
    
    ###### Step 4 - Solve the model and print some results
    # Solve the model
    print("\nSolving CP model....")
    msol = mdl.solve(FailLimit=1000000*len(Dags), RandomSeed = 1)
    #msol = mdl.solve(FailLimit=1000000*len(Dags), TemporalRelaxation = 'On')
    #msol = mdl.solve(FailLimit=2000000*len(Dags))
    #print("Completed")

    #print(msol.print_solution())
    #print(msol.is_solution_optimal())
    print(msol.get_objective_gaps())
    print(msol.get_objective_values())
    print(msol.get_objective_bounds())
    
    # This block of code, extract the CP solution into a list
    common.temp_list = [[] for i in range(len(Cluster))]
    for d in Dags:
        for f in Func_cluster:
            solution = msol.get_var_solution(pe_tasks[(d,f)])
            if solution.is_present():
                ID = task_names_ids[solution.get_name()]
                data = (ID, f[0], [solution.get_start(), solution.get_end()])
                #print(ID, f[0], solution.get_start(), solution.get_end())
                for i,name in enumerate(Cluster):
                    if f[0] == name:
                        common.temp_list[i].append(data)
    
    #print(common.temp_list)    
    common.PE_IDs = PE_IDs   
        
    if (common.simulation_mode == 'validation'):
        # plotting the results using visu library
        colors = ['salmon','turquoise', 'lime' , 'coral', 'lightpink']
        
        load = [CpoStepFunction() for j in range(len(Cluster))]
        for i in Dags:
            for f in Func_cluster:
                itv = msol.get_var_solution(pe_tasks[(d,f)])
                for j in range(len(Cluster)):
                    if itv.is_present() and Cluster[j] == f[0]:
                        load[j].add_value(itv.get_start(), itv.get_end(), 1)
                        #print(load[j].get_step_list(), j, f[1])
        
        for j in range(len(Cluster)):
            visu.panel(Cluster[j])
            visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, Capacity[j])], style='area', color='lightgrey')
            visu.function(segments=load[j], style='area', color=j)

        visu.panel("Tasks")
        for d in Dags:
            for t in Tasks:
                visu.interval(msol.get_var_solution(tasks[(d,t)]), t, tasks[(d,t)].get_name()) 
        visu.show()


# end of CP_cluster(......

###########################################################################################################################
###########################################################################################################################


def CP_PE(env_time, P_elems, resource_matrix, domain_applications, generated_jobs):
    ###### Step 1 - Initialize variable and parameters
    plt.close('all')
    num_of_tasks = len(generated_jobs[-1].task_list)

    
    # Set docplex related parameters
    context.solver.trace_log = False
    #params.CpoParameters.OptimalityTolerance = 2
    #params.CpoParameters.RelativeOptimalityTolerance= 2
    
    Dags = []
    # Get the task in Outstanding and Ready Queues
    # Since tasks in Completed Queue are already done, they will not be considered
    for task in common.TaskQueues.outstanding.list:
        if task.jobID not in Dags:
            Dags.append(task.jobID)
    for task in common.TaskQueues.ready.list:
        if task.jobID not in Dags:
            Dags.append(task.jobID)
    Dags.sort()
    #print(Dags)
    common.ilp_job_list = Dags


    NbDags = len(Dags)                                                      # Current number of jobs in the system
    PEs = []                                                                # List of PE in the given SoC configuration
    PE_IDs = []                                                             # List of IDs of PE in the given SoC configuration
    Tasks = []                                                              # list of tasks that CPLEX will return a schedule for
    Functionality = []                                                      # list of task-PE relationship
    Con_Precedence = []                                                     # list of task dependencies
    
    print('[I] Time %d: There is %d job(s) in the system' %(env_time, NbDags))
    print('[D] Time %d: ID of the jobs in the system are' %(env_time),Dags)
   
    
    ###### Step 2 - Prepare Data 
    
    # First, check if there are any tasks currently being executed on a PE
    # if yes, retrieve remaining time of execution on that PE and
    # do not assign a task during ILP solution
    for i, PE in enumerate(P_elems):
        if (PE.type == 'MEM') or (PE.type == 'CAC') :                       # Do not consider Memory ond Cache
            continue
        PEs.append(PE.name)                                                 # Populate names of the PEs in the SoC
        PE_IDs.append(PE.ID)                                                # Populate IDs of the PEs in the SoC

    #print(PEs)
    #print(PE_IDs)
    

    # This only supports streaming jobs from one application
    for task in generated_jobs[-1].task_list:
        if task.base_ID not in Tasks:
            Tasks.append(task.base_ID)                                      # Populate the base ID of the task 
            
        # Next, gather the information about which PE can run which Tasks 
        # and if so, the expected execution time
        for resource in resource_matrix.list:
            if (resource.type == 'MEM') or (resource.type == 'CAC') :        # Do not consider Memory ond Cache
                continue
            else:
                if task.name in resource.supported_functionalities:
                    ind = resource.supported_functionalities.index(task.name)
                    Functionality.append((resource.name, task.base_ID, resource.performance[ind]))

        
        # Finally, gather dependency information between tasks
        for i,predecessor in enumerate(task.predecessors):
            #print(task.ID, predecessor)
            for resource in resource_matrix.list:
                if (resource.type == 'MEM') or (resource.type == 'CAC') :
                    continue
                else:
                    pred_name = generated_jobs[-1].task_list[predecessor%num_of_tasks].name
                    #print(task.name, pred_name)
                    if (pred_name in resource.supported_functionalities):
                        #print(resource.name,Tasks[predecessor], task.name)
                        c_vol = generated_jobs[0].comm_vol[predecessor%num_of_tasks, task.base_ID]
                        #print(resource.name,Tasks[predecessor], task.name, c_vol)
                        Con_Precedence.append((resource.name,Tasks[predecessor%num_of_tasks], task.base_ID, c_vol))
    #print(len(Tasks))
    #print(Functionality)
    #print(Con_Precedence) 
    #print(len(Con_Precedence))
    

    
    ###### Step 3 - Create the model
    mdl = CpoModel()
    
      
    # Create dag interval variables
    dags = { d : mdl.interval_var(name="dag"+str(d)) for d in Dags}
    #print(dags)
    
    # Create tasks interval variables and pe_tasks interval variables
    # pe_tasks are optional and only one of them will be mapped to the corresponding
    # tasks interval variable
    # For example, task_1 has 3 pe_tasks (i.e.,P1-task_1, P2-task_1, P3, task_1)
    # only of these will be selected. If the first one is selected, it means that task_1 will be executed in P1
    tasks = {}
    pe_tasks = {}
    task_names_ids = {}
    for d in Dags:
        for i,t in enumerate(Tasks):
            tasks[(d,t)] = mdl.interval_var(name = str(d)+"_"+str(t)) 
        for f in Functionality:
            #print(f)
            name = str(d)+"_"+str(f[1])+'-'+str(f[0])
            
            if len(common.TaskQueues.running.list) == 0 and len(common.TaskQueues.completed.list) == 0:
                pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =int(f[2]), name = name)
                task_names_ids[name] = d*num_of_tasks+f[1]

            else:
                for ii, running_task in enumerate(common.TaskQueues.running.list):
                    if (d == running_task.jobID) and (f[1] == running_task.base_ID) and (f[0] == P_elems[running_task.PE_ID].name):

                        ind = resource_matrix.list[running_task.PE_ID].supported_functionalities.index(running_task.name)
                        exec_time = resource_matrix.list[running_task.PE_ID].performance[ind]
                        free_time = int(running_task.start_time + exec_time - env_time)
                        #print(free_time)    
                    
                        
                        pe_tasks[(d,f)] = mdl.interval_var(optional=True, start=0, end=free_time,name = name )
                        task_names_ids[name] = d*num_of_tasks+f[1]
                        break
                    elif (d == running_task.jobID) and (f[1] == running_task.base_ID):
                        pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =INTERVAL_MAX, name = name )
                        task_names_ids[name] = d*num_of_tasks+f[1]
                        break
                else:
                    pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =int(f[2]), name = name )
                    task_names_ids[name] = d*num_of_tasks+f[1]
                
                for iii, completed_task in enumerate(common.TaskQueues.completed.list):
                    if (d == completed_task.jobID) and (f[1] == completed_task.base_ID) and (f[0] == P_elems[completed_task.PE_ID].name):
                        #print(completed_task.name)
                        pe_tasks[(d,f)] = mdl.interval_var(optional=True, start=0, end=0, name = name )
                        task_names_ids[name] = d*num_of_tasks+f[1]
                    elif (d == completed_task.jobID) and (f[1] == completed_task.base_ID):
                        pe_tasks[(d,f)] = mdl.interval_var(optional=True, size =INTERVAL_MAX, name = name )
                        task_names_ids[name] = d*num_of_tasks+f[1]

                
    #print(tasks)                    
    #print((pe_tasks))
    #print(task_names_ids)
    
    
    # Add the temporal constraints    
    for d in Dags:
        for c in Con_Precedence:
            #print (c)
            for (p1, task1, d1) in Functionality:
                if p1 == c[0] and task1 == c[1]:
                    p1_id = PEs.index(p1)
                    #print(p1_id)
                    for (p2, task2, d2) in Functionality:
                        if p2 == c[0] and task2 == c[2]:
                            #print (d,p1,p2, task1,task2, 0)
                            #print(pe_tasks[d,(p1,task1,d1)])
                            #print(pe_tasks[d,(p2,task2,d2)])
                            mdl.add( mdl.end_before_start(pe_tasks[d,(p1,task1,d1)], pe_tasks[d,(p2,task2,d2)], 0) )
                        elif p2 != c[0] and task2 == c[2]:
                            p2_id = PEs.index(p2)
                            #print(p2_id)
                            bandwidth = common.ResourceManager.comm_band[p1_id,p2_id]
                            comm_time = int( (c[3])/bandwidth )
                            for ii, completed_task in enumerate(common.TaskQueues.completed.list):
                                
                                if ((d == completed_task.jobID) and (task1 == completed_task.base_ID) and (p1 == P_elems[completed_task.PE_ID].name)):
                                    mdl.add( mdl.end_before_start(pe_tasks[d,(p1,task1,d1)], pe_tasks[d,(p2,task2,d2)], max(0,comm_time+completed_task.finish_time-env_time)  ))
                                    #print (d, p1,p2, task1,task2, max(0,int(c[3])+completed_task.finish_time-env_time) )
                                    break
                            else:
                                #print(d, p1,p2, task1,task2, c[3])
                                mdl.add( mdl.end_before_start(pe_tasks[d,(p1,task1,d1)], pe_tasks[d,(p2,task2,d2)], comm_time ) )
    

    # Add end of dags constraints
    for i,d in enumerate(Dags):
        for j in range(i):
            #print(j,i)
            mdl.add ( mdl.end_of(dags[Dags[j]]) <= mdl.end_of(dags[Dags[i]]) )
            #mdl.add ( mdl.start_of(dags[Dags[j]]) <= mdl.start_of(dags[Dags[i]]) )
    
    # Add the span constraints
    # This constraint enables to identify tasks in a dag
    for d in Dags:
        mdl.add( mdl.span(dags[d], [tasks[(d,t)] for t in Tasks ] ) )
    
    
    # Add the alternative constraints
    # This constraint ensures that only one PE is chosen to execute a particular task
    for d in Dags:
        for t in Tasks:
            mdl.add( mdl.alternative(tasks[d,t], [pe_tasks[d,f] for f in Functionality if f[1]==t]) )
            
    for p in PEs:   
        #mdl.add( mdl.no_overlap(processing_elements[p]))
        a_list = [pe_tasks[d,f] for d in Dags for f in Functionality if f[0]==p]
        #print([pe_tasks[f] for f in Functionality if f[0]==p])
        if a_list:
            mdl.add( mdl.no_overlap([pe_tasks[d,f] for d in Dags for f in Functionality if f[0]==p]))
        else:
            continue        
    
     
    # Add the objective
    #mdl.add(mdl.minimize(mdl.sum([mdl.end_of(dags[d])*(i+1) for i,d in enumerate(Dags)])))
    #mdl.add(mdl.minimize(mdl.sum([mdl.end_of(dags[d])*(len(Dags)-i) for i,d in enumerate(Dags)])))
    mdl.add(mdl.minimize(mdl.sum([mdl.end_of(dags[d]) for i,d in enumerate(Dags)])))
    #mdl.add(mdl.minimize(mdl.max([mdl.end_of(pe_tasks[(d,f)]) for i,d in enumerate(Dags) for f in Functionality])))
    

    
    ###### Step 4 - Solve the model and print some results
    # Solve the model
    print("\nSolving CP model....")
    #msol = mdl.solve(FailLimit=1000000*len(Dags), RandomSeed = 1)
    #msol = mdl.solve(FailLimit=1000000*len(Dags), TemporalRelaxation = 'On')
    #msol = mdl.solve(FailLimit=1000000*len(Dags), TimeLimit = 60)
    msol = mdl.solve(TimeLimit = 60)
    #print("Completed")

    #print(msol.is_solution_optimal())
    print(msol.get_objective_gaps())
        
    
    tem_list = []
    for d in Dags:
        for f in Functionality:
            solution = msol.get_var_solution(pe_tasks[(d,f)])
            #print(solution)
            if solution.is_present():
                ID = task_names_ids[solution.get_name()]
                tem_list.append( (ID, f[0], solution.get_start(), solution.get_end()) )
                #print(ID, f[0], solution.get_start(), solution.get_end())
    tem_list.sort(key=lambda x: x[2], reverse=False)
    #print(tem_list)
    
    actual_schedule = []
    for i,p in enumerate(PEs):
        count = 0      
        for item in tem_list:
            if item[1] == p:
                actual_schedule.append( (item[0],i,count+1))
                count += 1
    actual_schedule.sort(key=lambda x: x[0], reverse=False)

    common.table = []
    for element in actual_schedule:
        common.table.append((element[1],element[2]))
    #print(common.table)        
    

    
    if (common.simulation_mode == 'validation'):
        colors = ['salmon','turquoise', 'lime' , 'coral', 'lightpink']
        #colors = ['salmon','turquoise', 'lime' , 'coral', 'lightpink', 'silver', 'slateblue', 'plum', 'lavender','bisque']
        #colors = ['red', 'blue', 'green', 'brown', 'magenta']
        PEs.reverse()                     
        for i,p in enumerate(PEs):
            #visu.panel()
            #visu.pause(PE_busy_times[p])
            visu.sequence(name=p)

            for ii,d in enumerate(Dags):
                for f in Functionality:  
                    wt = msol.get_var_solution(pe_tasks[(d,f)])
                    if wt.is_present() and p == f[0]:
                        color = colors[ii%len(colors)]
                        visu.interval(wt, color, str(task_names_ids[wt.get_name()]))
                        #print(t.name)

        visu.show()

    for d in Dags:
       for task in generated_jobs[d].task_list:
           task.dynamic_dependencies.clear()                                     # Clear dependencies from previosu ILP run
           #num_of_tasks = len(generated_jobs[d].task_list)
           ind = Dags.index(task.jobID)
           task_sched_ID = task.base_ID + ind*num_of_tasks        
           task_order = common.table[task_sched_ID][1]
           for k in Dags:
               for dyn_depend in generated_jobs[k].task_list:
                   dyn_depend_ind = Dags.index(dyn_depend.jobID)
                   dyn_depend_sched_ID = dyn_depend.base_ID + dyn_depend_ind*num_of_tasks

                   if ( (common.table[dyn_depend_sched_ID][0] == common.table[task_sched_ID][0]) and 
                        (common.table[dyn_depend_sched_ID][1] == task_order-1) and 
                        (dyn_depend.ID not in task.predecessors) and 
                        (dyn_depend.ID not in task.dynamic_dependencies) ):
                       #print([m.ID != task1.ID for task1 in common.TaskQueues.completed.list])
                       task.dynamic_dependencies.append(dyn_depend.ID)

           #print(task.ID, task.dynamic_dependencies)
# end of def CP_PE

###########################################################################################
###########################################################################################

