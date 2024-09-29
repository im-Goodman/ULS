'''
Description: This file contains the DVFS policies.
'''

import sys
import math
import csv
import os
import ast
from math import exp
import numpy as np

import common
import DTPM_power_models
import DASH_Sim_utils
import DTPM_utils
import DTPM_run_dagger

#########################################################################################
## DYPO Coefficients
#########################################################################################
##--------------------------------------------------------------------Dypo--------------------------------------------------------------------
Beta = [
    [-12.83413609,	0.169871649,	-9.20304029,	0.881841519,	0.161333916,	6.081802598,	0.081705975,	0.129381729,	-2.313895343,	-0.025914689,	0.471669711],
    [29.84384143,	0.077145311,	-5.780148299,	3.390355289,	0.09004552,	-3.468713328,	0.05979893,	0.159562879,	-1.975354257,	-0.296241105,	-0.138877444],
]

Beta_1 = [
    [377.6180336,	-1.63E-02,	-5.68E+01,	-6.09E+01,	0.69788545,	-60.58109058,	2.010745185,	2.663100721, 4.41E-01,	0.672579894,	-1.467944771],
    [-348.2430533,	2.25E+00,	-1.17E+02,	4.01E+01,	3.69606839, -277.1006487,	-1.791934134,	-1.499709793,	1.200984169,	-10.46190386,	-0.060283622],
]
##--------------------------------------------------------------------Dypo--------------------------------------------------------------------

def initialize_frequency(cluster):
    if cluster.current_frequency == 0:
        if cluster.DVFS == 'ondemand':
            cluster.current_frequency = DTPM_power_models.get_max_freq(cluster.OPP)
            cluster.current_voltage = DTPM_power_models.get_max_voltage(cluster.OPP)
            cluster.policy_frequency = DTPM_power_models.get_max_freq(cluster.OPP)
        elif cluster.DVFS == 'performance':
            cluster.current_frequency = DTPM_power_models.get_max_freq(cluster.OPP)
            cluster.current_voltage = DTPM_power_models.get_max_voltage(cluster.OPP)
            cluster.policy_frequency = DTPM_power_models.get_max_freq(cluster.OPP)
        elif cluster.DVFS == 'powersave':
            cluster.current_frequency = DTPM_power_models.get_min_freq(cluster.OPP)
            cluster.current_voltage = DTPM_power_models.get_min_voltage(cluster.OPP)
            cluster.policy_frequency = DTPM_power_models.get_min_freq(cluster.OPP)
        elif str(cluster.DVFS).startswith('constant'):
            DVFS_str_split = str(cluster.DVFS).split("-")
            constantFrequency = int(DVFS_str_split[1])
            cluster.current_frequency = constantFrequency
            cluster.current_voltage = DTPM_power_models.get_voltage_constant_mode(cluster.OPP, constantFrequency)
            cluster.policy_frequency = constantFrequency
        elif cluster.DVFS == 'imitation-learning':
            middle_opp = math.floor(len(cluster.OPP) / 2)
            cluster.current_frequency  = cluster.OPP[middle_opp][0]
            cluster.current_voltage    = cluster.OPP[middle_opp][1]
            cluster.policy_frequency = cluster.OPP[middle_opp][0]
        elif cluster.DVFS == 'DyPO' :
            cluster.current_frequency = DTPM_power_models.get_max_freq(cluster.OPP)
            cluster.current_voltage = DTPM_power_models.get_max_voltage(cluster.OPP)
            cluster.policy_frequency = DTPM_power_models.get_max_freq(cluster.OPP)

def ondemand_policy(cluster, PEs, timestamp):
    # When using ondemand, evaluate the PE utilization and adjust the frequency accordingly
    utilization = DASH_Sim_utils.get_cluster_utilization(cluster, PEs) * cluster.num_active_cores
    if utilization <= common.util_high_threshold and utilization >= common.util_low_threshold:
        # Keep the current frequency
        DTPM_power_models.keep_frequency(cluster, timestamp)
    elif utilization > common.util_high_threshold:
        # Only modify the frequency if the cluster is not being throttled
        if common.throttling_state == -1:
            # Set the maximum frequency
            DTPM_power_models.set_max_frequency(cluster, timestamp)
    elif utilization < common.util_low_threshold:
        # Decrease the frequency
        DTPM_power_models.decrease_frequency(cluster, timestamp)
    else:
        print("[E] Error while evaluating the PE utilization in the DVFS module, all test cases must be previously covered")
        sys.exit()
    cluster.policy_frequency = cluster.current_frequency

array = []
def cmp(a, b) :
    if array[0] < array[1] :
        return -1
    else :
        return 1
# def cmp(a, b) :   
        
def DyPO_policy(resource_matrix, PEs, timestamp) :

    NORM_PARA    = 100
    NUM_CLASS    = 3
    NUM_FEATURES = 11
    NUM_THREADS  = 1
    
    system_state = DASH_Sim_utils.get_system_state(PEs)
    ## system_state = system_state.drop(['Job List', 'Utilization_PE_0', 'Utilization_PE_1'])

    ## Assemble features
    cpu4_freq = system_state['FREQ_PE_1 (GHz)']
    cpu5_freq = system_state['FREQ_PE_1 (GHz)']
    cpu4_util_value = system_state['Utilization_PE_1'] * 100
    cpu5_util_value = system_state['Utilization_PE_1'] * 100
    values = []
    values.append(1)
    values.append(system_state['CPU Cycles'])
    values.append(system_state['Branch Miss Prediction'])
    values.append(system_state['Level 2 cache misses'])
    values.append(system_state['Data Memory Access'])
    values.append(system_state['Non-cache External Mem. Req.'])

    tid = 0

    if tid == 1 or tid == 0 :

        curr_freq_test = cpu4_freq * 1000
        if (curr_freq_test == 0) :
            curr_freq_test = cpu5_freq * 1000
        ## if (curr_freq_test == 0) :
        
        common.DyPO_dypo_curr_freq = curr_freq_test

        ## if (common.DyPO_dypo_prev_freq == 0) :
        ##     common.DyPO_dypo_prev_freq = curr_freq_test
        ## ## if (common.DyPO_dypo_prev_freq == 0) :
        
        features   = np.zeros((NUM_FEATURES,))
        H          = np.zeros((NUM_CLASS,))
        Prob       = np.zeros((NUM_CLASS,))
        Prob_final = 1
        H_total    = 1
        exp_H_val  = np.zeros((2,))
        
        total_util_a7 = system_state['Utilization_PE_0']
        
        ##--------------------------------------------------Sort core utilization-------------------------------------------------------------------
        
        core_util_a15 = np.zeros((2,))
        core_util_a15[0] = cpu4_util_value
        core_util_a15[1] = cpu5_util_value

        ## Sort the core utilization
        size_a15 = 2
        index_a15 = np.zeros((size_a15,), dtype=np.int32)
        
        for i in range(size_a15) :
            index_a15[i] = i
        ## for i in range(size_a15) :

        array = core_util_a15
        ## qsort(index_a15, size_a15, sizeof(*index_a15), cmp)
#        array.sort(kind='quicksort', cmp=cmp)
        
        NUM_A15 = 4
        a15_decision = index_a15[1] + NUM_A15
        
        ##printf("\n UTIL A7: %f,%f,%f  \n", core_util_a7[0],core_util_a7[1],core_util_a7[2]);
        ##printf("\n UTIL A15: %f,%f,%f,%f  \n", core_util_a15[0],core_util_a15[1],core_util_a15[2],core_util_a15[3]);
        ##printf("\n UTIL A15: %f,%f,%f,%f  \n", core_util_a15[index_a15[0]],core_util_a15[index_a15[1]],core_util_a15[index_a15[2]],core_util_a15[index_a15[3]]);
        ##--------------------------------------------------End of Sort core utilization-------------------------------------------------------------------
         
        ##---------------------------Capture features----------------------------------------

        for i in range(NUM_FEATURES) :
            if (i==0) :
                features[i] = 1
            elif (i==1) :
                features[i] = (1.0*values[1]/values[0])*NORM_PARA*NUM_THREADS
            elif (i==2) :
                features[i] = (1.0*values[2]/values[0])*NORM_PARA*NUM_THREADS
            elif (i==3) :
                features[i] = (1.0*values[3]/values[0])*NORM_PARA*NUM_THREADS
            elif (i==4) :
                features[i] = (1.0*values[4]/values[0])*NORM_PARA*NUM_THREADS
            elif (i==5) :
                features[i] = (1.0*values[5]/values[0])*NORM_PARA*NUM_THREADS
            elif (i==6) :
                features[i] = total_util_a7
            elif (i==7) :
                features[i] = core_util_a15[index_a15[1]]				## Highest utilization of A15
            elif (i==8) :
                features[i] = core_util_a15[index_a15[0]]
            elif (i==9) :
                features[i] = 0.0 ##core_util_a15[index_a15[1]]
            elif (i==10) :
                features[i] = 0.0 ##core_util_a15[index_a15[0]];				## Lowest utilization of A15
        ## for i in range(NUM_FEATURES) :
        
        ##---------------------------LEVEL 1--------------------------------
        
        ##---------------------------Calculate H using Beta and features----------------------------------------
        for i in range(NUM_CLASS -1) :
            for j in range(NUM_FEATURES) :
                H[i] += Beta[i][j] * features[j]
            ##	printf("\n--Beta Values: %f--",Beta[i][j])
            ##	printf("\n--Features : %f--",features[j])
            ##	printf("\n--H Values: %f--\n",H[i])
            ## for j in range(NUM_FEATURES) :
        ## for i in range(NUM_CLASS -1) :
        
        for i in range(NUM_CLASS -1) :
            exp_H_val[i] =  exp(H[i])
            H_total      += exp_H_val[i]
        ## for (i=0; i < (NUM_CLASS -1); i++){
        
        for i in range(NUM_CLASS -1) :
            Prob[i]    = exp_H_val[i] / H_total
            Prob_final = Prob_final - Prob[i]
        ## for (i=0; i < (NUM_CLASS -1); i++){
        
        ##---------------------------Sort probability-------------------------------------------------------------

        max   = 0
        index = 0
        for i in range(NUM_CLASS -1) :
            if (max < Prob[i]) :
                max   = Prob[i]
                index = i
            ## if (max < Prob[i]) :
        ## for(i=0;i< (NUM_CLASS-1);i++){
        
        ## Find max of all probability
        if (max < Prob_final) :
            max   = Prob_final
            index = NUM_CLASS-1
        ## if (max < Prob_final) :
        
        features     = np.zeros((NUM_FEATURES,))
        H_1          = np.zeros((NUM_CLASS,))
        Prob_1       = np.zeros((NUM_CLASS,))
        Prob_final_1 = 1
        H_total_1    = 1
        exp_H_val_1  = np.zeros((2,))
        max_1        = 0
        index_1      = 0
        
        if (index == 1) :
            ##---------------------------Calculate H using Beta and features----------------------------------------
            for i in range(NUM_CLASS - 1) :
                for j in range(NUM_FEATURES) :
                    H_1[i] += Beta_1[i][j] * features[j]
                ##	printf("\n--Beta Values: %f--",Beta[i][j])
                ##	printf("\n--Features : %f--",features[j])
                ##	printf("\n--H Values: %f--\n",H[i])
                ## for (j=0; j < NUM_FEATURES; j++){
            ## for i in range(NUM_CLASS - 1) :

            for i in range(NUM_CLASS - 1) :
                exp_H_val_1[i] =  exp(H_1[i])
                H_total_1      += exp_H_val_1[i]
                ##printf("\n--H Total_1: %Lf--\n",H_total_1)
            ## for i in range(NUM_CLASS - 1) :

            for i in range(NUM_CLASS - 1) :
                Prob_1[i]    = exp_H_val_1[i]/H_total_1
                Prob_final_1 = Prob_final_1 - Prob_1[i]
            ## for i in range(NUM_CLASS - 1) :
        
            ##---------------------------Sort probability-------------------------------------------------------------
        
            for i in range(NUM_CLASS - 1) :
                if (max_1 < Prob_1[i]) :
                    max_1   = Prob_1[i]
                    index_1 = i
                ## if (max_1 < Prob_1[i])
            ## for i in range(NUM_CLASS - 1) :
        
            if (max_1 < Prob_final_1) :
                max_1   = Prob_final_1
                index_1 = NUM_CLASS-1
            ## if (max_1 < Prob_final_1){
        ## if (index == 1) :
        
        ##---------------------------Apple Classifier decision------------------------------------------------------------
        if (common.DyPO_prev_index != index) or (common.DyPO_prev_index_1 != index_1) :
            common.DyPO_STATUS   = 0
            common.DyPO_F_STATUS = 0
        ## if (prev_index != index) or (prev_index_1 != index_1) :
        
        if ((index != 1 and common.DyPO_prev_index   == index   and common.DyPO_STATUS == 1) or
                (index == 1 and common.DyPO_prev_index_1 == index_1 and common.DyPO_STATUS == 1)) :
            a = 0
            ## Decision already taken

        elif (index == 0) :

            ## Decision for A7
            common.DyPO_dypo_curr_freq = 800

            ## Decision for A15
            common.DyPO_dypo_req_big = 1

            if ( (core_util_a15[index_a15[0]] == 0) and (core_util_a15[index_a15[1]] < 80) ) :
                if (common.DyPO_dypo_req_big != common.DyPO_dypo_n_big) :
                    ## change_online_offline_cores(index_a15[0] + NUM_A15,0)
                    # common.ClusterManager.cluster_list[1].num_active_cores = common.DyPO_dypo_req_big
                    DTPM_power_models.set_active_cores(common.ClusterManager.cluster_list[1], PEs, common.DyPO_dypo_req_big)
                    common.DyPO_dypo_n_big -= 1
                ## if (dypo_req_big != dypo_n_big) :
                common.DyPO_STATUS = 1
            ## if ( (core_util_a15[index_a15[0]] == 0) and (core_util_a15[index_a15[1]] < 80) ) :

            ## Rest utilization index
            for i in range(size_a15) :
                index_a15[i] = 0
            ## for i in RANGE(size_a15) :

            input_freq = []
            for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                if current_cluster.type != "MEM":
                    input_freq.append(common.DyPO_dypo_curr_freq / 1000)
            DTPM_power_models.set_frequency(timestamp, input_freq, False)

        elif (index == 2) :

            ## Decision for A7
            common.DyPO_dypo_curr_freq = 1000

            ## Decision for A15
            common.DyPO_dypo_req_big = 1
        
            if ( (core_util_a15[index_a15[0]] == 0.0) and (core_util_a15[index_a15[1]] < 80.0) ) :
                if (common.DyPO_dypo_req_big != common.DyPO_dypo_n_big) :
                    #change_online_offline_cores(index_a15[0] + NUM_A15, 0)
                    # common.ClusterManager.cluster_list[1].num_active_cores = common.DyPO_dypo_req_big
                    DTPM_power_models.set_active_cores(common.ClusterManager.cluster_list[1], PEs, common.DyPO_dypo_req_big)
                    common.DyPO_dypo_n_big -= 1
                ## if (common.DyPO_dypo_req_big != common.DyPO_dypo_n_big) :
                common.DyPO_STATUS = 1
            ## if ( (core_util_a15[index_a15[0]] == 0.0) and (core_util_a15[index_a15[1]] < 80.0) ) :
                    
            ## Rest utilization index
            for i in range(size_a15) :
                index_a15[i] = 0
            ## for i in range(size_a15) :
        
            input_freq = []
            for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                if current_cluster.type != "MEM":
                    input_freq.append(common.DyPO_dypo_curr_freq / 1000)
            DTPM_power_models.set_frequency(timestamp, input_freq, False)

        elif ((index == 1 ) and ((index_1 == 0) or (index_1 == 2))) :
        
            common.DyPO_dypo_curr_freq = 800
                    
            ## Decision for A15
            ## Change A15 frequency
            common.DyPO_dypo_req_big = 2

            if ( (core_util_a15[index_a15[0]] == 0.0) and (core_util_a15[index_a15[1]] < 80.0) ) :
                if (common.DyPO_dypo_req_big != common.DyPO_dypo_n_big) :
                    #change_online_offline_cores(index_a15[0] + NUM_A15,1)
                    # common.ClusterManager.cluster_list[1].num_active_cores = common.DyPO_dypo_req_big
                    DTPM_power_models.set_active_cores(common.ClusterManager.cluster_list[1], PEs, common.DyPO_dypo_req_big)
                    common.DyPO_dypo_n_big += 1
                ## if (common.DyPO_dypo_req_big != common.DyPO_dypo_n_big) :
                common.DyPO_STATUS = 1
            ## if ( (core_util_a15[index_a15[0]] == 0.0) and (core_util_a15[index_a15[1]] < 80.0) ) :

            ## Rest utilization index
            for i in range(size_a15) :
                index_a15[i] = 0
            ## for i in range(size_a15) :
        
            input_freq = []
            for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                if current_cluster.type != "MEM":
                    input_freq.append(common.DyPO_dypo_curr_freq / 1000)
            DTPM_power_models.set_frequency(timestamp, input_freq, False)

        elif (index == 1 and index_1 == 1) :

            ## Decision for A7
            common.DyPO_dypo_curr_freq = 800

            common.DyPO_dypo_req_big = 1

            if ( (core_util_a15[index_a15[0]] == 0.0) and (core_util_a15[index_a15[1]] < 80.0) ) :
                if (common.DyPO_dypo_req_big != common.DyPO_dypo_n_big) :
                    #change_online_offline_cores(index_a15[0] + NUM_A15,0)
                    # common.ClusterManager.cluster_list[1].num_active_cores = common.DyPO_dypo_req_big
                    DTPM_power_models.set_active_cores(common.ClusterManager.cluster_list[1], PEs, common.DyPO_dypo_req_big)
                    common.DyPO_dypo_n_big -= 1
                ## if (common.DyPO_dypo_req_big != common.DyPO_dypo_n_big) :
                common.DyPO_STATUS = 1
            ## if ( (core_util_a15[index_a15[0]] == 0.0) and (core_util_a15[index_a15[1]] < 80.0) ) :

            ## Rest utilization index
            for i in range(size_a15) :
                index_a15[i] = 0
            ## for i in range(size_a15) :
        
            input_freq = []
            for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
                if current_cluster.type != "MEM":
                    input_freq.append(common.DyPO_dypo_curr_freq / 1000)
            DTPM_power_models.set_frequency(timestamp, input_freq, False)

        ## if ((index != 1 and prev_index   == index   and STATUS == 1) or \
        ##	(index == 1 and prev_index_1 == index_1 and STATUS == 1)) :
        
        common.DyPO_prev_index     = index
        common.DyPO_prev_index_1   = index_1
        common.DyPO_dypo_prev_freq = common.DyPO_dypo_curr_freq

        DASH_Sim_utils.trace_IL_predictions(timestamp, [common.DyPO_dypo_curr_freq] * 2, [common.ClusterManager.cluster_list[0].num_active_cores, common.ClusterManager.cluster_list[1].num_active_cores])


def imitation_learning_policy(resource_matrix, PEs, timestamp):
    if common.IL_freq_policy == None:
        print("[E] Error while loading the IL policy for frequencies")
        sys.exit()
    if common.IL_num_cores_policy == None and common.enable_num_cores_prediction:
        print("[E] Error while loading the IL policy for number of cores")
        sys.exit()
    if common.IL_regression_policy == None and common.enable_regression_policy:
        print("[E] Error while loading the IL regression policy for execution time")
        sys.exit()
    if not os.path.exists(common.HARDWARE_COUNTERS_TRACE):
        print("[E] Hardware counters file was not found. This file is created by generate_traces.py and it is needed for IL policies.")
        sys.exit()

    system_state = DASH_Sim_utils.get_system_state(PEs)
    system_state = system_state.drop(['Job List', 'Utilization_PE_0', 'Utilization_PE_1'])

    oracle_row = common.oracle_config_dict[str(common.current_job_list)]

    common.total_predictions += 1

    # -- Frequency --
    # Prediction
    predicted_frequency = ast.literal_eval(common.IL_freq_policy.predict([system_state])[0])
    # Set frequency
    if common.throttling_state != -1:
        input_freq = []
        for idx, current_cluster in enumerate(common.ClusterManager.cluster_list):
            if current_cluster.type != "MEM":
                input_freq.append(min(current_cluster.current_frequency / 1000, predicted_frequency[idx]))
        DTPM_power_models.set_frequency(timestamp, input_freq, True)
    else:
        DTPM_power_models.set_frequency(timestamp, predicted_frequency, False)
    # Get oracle
    oracle_freq = []
    for cluster_ID in range(len(common.ClusterManager.cluster_list) - 1):
        oracle_freq.append(oracle_row['FREQ_PE_' + str(cluster_ID) + ' (GHz)'])
    # Check the prediction
    if predicted_frequency != oracle_freq:
        common.wrong_predictions_freq += 1
        if common.snippet_ID_exec != common.DAgger_last_snippet_ID_freq:
            # Add only one sample/snippet when the prediction is wrong
            common.DAgger_last_snippet_ID_freq = common.snippet_ID_exec
            # Data will be aggregated when this snippet finishes (inside create_dataset_IL_DTPM method)
            common.aggregate_data_freq = True

    # -- Number of active cores --
    if common.enable_num_cores_prediction:
        # Prediction
        predicted_num_cores = ast.literal_eval(common.IL_num_cores_policy.predict([system_state])[0])
        # Set number of cores
        for current_cluster in common.ClusterManager.cluster_list:
            if current_cluster.type == "LTL":
                # current_cluster.num_active_cores = predicted_num_cores[0]
                DTPM_power_models.set_active_cores(current_cluster, PEs, predicted_num_cores[0])
            elif current_cluster.type == "BIG":
                # current_cluster.num_active_cores = predicted_num_cores[1]
                DTPM_power_models.set_active_cores(current_cluster, PEs, predicted_num_cores[1])
        # Get oracle
        oracle_num_cores = []
        oracle_num_cores.append(oracle_row['N_little'])
        oracle_num_cores.append(oracle_row['N_big'])
        # Check the prediction
        if predicted_num_cores != oracle_num_cores:
            common.wrong_predictions_num_cores += 1
            if common.snippet_ID_exec != common.DAgger_last_snippet_ID_num_cores:
                # Add only one sample/snippet when the prediction is wrong
                common.DAgger_last_snippet_ID_num_cores = common.snippet_ID_exec
                # Data will be aggregated when this snippet finishes (inside create_dataset_IL_DTPM method)
                common.aggregate_data_num_cores = True
        DASH_Sim_utils.trace_IL_predictions(timestamp, predicted_frequency, predicted_num_cores)
    else:
        DASH_Sim_utils.trace_IL_predictions(timestamp, predicted_frequency)

    # -- Regression --
    if common.enable_regression_policy:
        # Restore previous freq for regression
        if len(common.previous_freq) == 2:
            predicted_frequency = common.previous_freq[0]
            common.previous_freq.pop(0)
        common.previous_freq.append(predicted_frequency)
        deadline_met = False
        max_freq_flag = False
        deadline = common.deadline_dict[str(common.current_job_list)]
        system_state = DASH_Sim_utils.get_system_state(PEs, input_freq=predicted_frequency)
        system_state = system_state.drop(['Job List', 'Utilization_PE_0', 'Utilization_PE_1'])
        # Prediction
        predicted_exec_time = common.IL_regression_policy.predict([system_state])[0]
        # Check if the deadline can be met
        if predicted_exec_time > deadline:
            max_freq_flag, predicted_frequency = DTPM_power_models.increase_frequency_all_PEs(predicted_frequency)
        else:
            deadline_met = True
        # Set frequency
        DTPM_power_models.set_frequency(timestamp, predicted_frequency, False)

        if common.enable_num_cores_prediction:
            # Restore previous num_cores for regression
            if len(common.previous_num_cores) == 2:
                predicted_num_cores = common.previous_num_cores[0]
                common.previous_num_cores.pop(0)
            common.previous_num_cores.append(predicted_num_cores)
            if not deadline_met and max_freq_flag:
                system_state = DASH_Sim_utils.get_system_state(PEs, input_cores=predicted_num_cores)
                system_state = system_state.drop(['Job List', 'Utilization_PE_0', 'Utilization_PE_1'])
                # Prediction
                predicted_exec_time = common.IL_regression_policy.predict([system_state])[0]
                # Check if the deadline can be met
                if predicted_exec_time > deadline:
                    max_num_cores_flag, predicted_num_cores = DTPM_power_models.increase_num_cores_all_PEs(predicted_num_cores)
                # Set number of cores
                for current_cluster in common.ClusterManager.cluster_list:
                    if current_cluster.type == "LTL":
                        # current_cluster.num_active_cores = predicted_num_cores[0]
                        DTPM_power_models.set_active_cores(current_cluster, PEs, predicted_num_cores[0])
                    elif current_cluster.type == "BIG":
                        # current_cluster.num_active_cores = predicted_num_cores[1]
                        DTPM_power_models.set_active_cores(current_cluster, PEs, predicted_num_cores[1])

        if common.snippet_ID_exec != common.DAgger_last_snippet_ID_regression:
            # Add only one sample/snippet when the prediction is wrong
            common.DAgger_last_snippet_ID_regression = common.snippet_ID_exec
            # Data will be aggregated when this snippet finishes (inside create_dataset_IL_DTPM method)
            common.aggregate_data_regression = True

        if (common.DEBUG_SIM):
            print('[D] Time %d: The DVFS policy predicted: %d' % (timestamp, predicted_frequency))


def aggregate_data(dataset_name, system_state, oracle):
    # Aggregate sample to the dataset
    if common.train_on_reduced_dataset:
        dataset_file = common.DATASET_FILE_DTPM.split('.')[0] + "_" + dataset_name + "_oracle_reduced.csv"
    else:
        dataset_file = common.DATASET_FILE_DTPM.split('.')[0] + "_" + dataset_name + "_oracle.csv"
    with open(dataset_file, 'a', newline='') as csvfile:
        dataset = csv.writer(csvfile, delimiter=',')
        system_state['Oracle_' + str(dataset_name)] = str(oracle)
        dataset.writerow(system_state)

