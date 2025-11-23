# -*- coding: utf-8 -*-
from __future__ import print_function
import wntr
import numpy as np
import os
#import csv
import pandas as pd#(matheus)import pandas
import time
import matplotlib.pyplot as plt
from wntr.epanet.util import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

#import sys

# Set duration in hours
sim_step_minutes = 30
durationHours = 24*365 # One Year
timeStamp = pd.date_range("2024-01-01 00:00", "2024-12-30 23:55", freq=str(sim_step_minutes)+"min")

DATASET_FOLDER_NAME = "EPANET Net 3_T_OD"
WDSName = "EPANET Net 3"

df_results = pd.read_csv(DATASET_FOLDER_NAME+'/Summary.csv', header=0, index_col=0)

print(["Run input file: ", DATASET_FOLDER_NAME])

INP_UNITS = FlowUnits.LPS

dataset_folder_os = os.getcwd()+'\\'+DATASET_FOLDER_NAME

plot_graphs=False

results_decimal_digits = 5

# RUN SCENARIOS
def runScenarios(scNum):
    print('Running scenario: '+str(scNum))
    
    #Define paths
    sc_folder_os = dataset_folder_os+'\\Scenario-'+str(scNum)

    OD_folder_wntr = DATASET_FOLDER_NAME + '/Scenario-'+str(scNum)
    OD_inp_file_wntr = OD_folder_wntr + '/' + WDSName +'_OnlyDemands-'+str(scNum) + '.inp'

    ODdem = pd.read_csv(OD_folder_wntr +'/OD_Node_demands.csv', index_col=[0])
    ODdem.index = pd.to_datetime(ODdem.index)

    ODest_node_press = pd.read_csv(OD_folder_wntr +'/OD_Estimated_Node pressures.csv', index_col=[0])
    ODest_node_press.index = pd.to_datetime(ODest_node_press.index)

    ODest_link_flows = pd.read_csv(OD_folder_wntr +'/OD_Estimated_Link_flows.csv', index_col=[0])
    ODest_link_flows.index = pd.to_datetime(ODest_link_flows.index)

    #this is just for tests considering  real leak demand, but leak demand isn't actually an available information
    real_leak_demand0 = pd.read_csv(OD_folder_wntr +'/Leaks/Leak_leak_node0_demand.csv', index_col=[0])
    real_leak_demand0.index = pd.to_datetime(real_leak_demand0.index)

    wn = wntr.network.WaterNetworkModel(OD_inp_file_wntr)

    #Calculate loss signal
    loss = - ODdem.sum(axis=1)

    #Graph plots for Master's Thesis - Example of indirect pressure measurement, based on REAL PRESSURE measurements calibration
    if scNum==1 and plot_graphs==True:
        fig = plt.figure(1)
        ax = fig.add_subplot(3,1,1)
        ax.set_title("(a) - Customer Node 157 without Particular Tank")
        ax.set_ylabel("Demand (L/s)")
        xvals = np.arange(286)*0.5
        ax.plot(xvals, ODdem['157'].iloc[0:len(xvals)], color='red')

        ax = fig.add_subplot(3,1,2)
        ax.set_title("(b) - Customer Node 153 with Particular Tank)")
        ax.set_ylabel("Demand (L/s)")
        ax.plot(xvals, ODdem['153'].iloc[0:len(xvals)], color='green')
        
        ax = fig.add_subplot(3,1,3)
        ax.set_title("(c) - Customer Node 153 with Particular Tank)")
        ax.set_ylabel("Pressure head (m)")
        ax.plot(xvals, ODest_node_press['153'].iloc[0:len(xvals)], color='blue', label='Measured')
        ax.plot(xvals, 8.01612919027507092844 + 0.044500607749219091781*ODdem['153'].iloc[0:len(xvals)]**2 + 4.1212704255141389486*ODdem['153'].iloc[0:len(xvals)]**1.85, color='black', label='Infered from demand', linestyle='dashed')
        ax.set_xlabel("Time (hours)")
        ax.legend(loc='upper right')

        plt.show()

    #Leak Detection and Localization Parameters and Variables
    init_big_value = 100000000
    historical_min_loss=init_big_value
    num_complete_days_before_leak = 4
    initial_historical_period = round(num_complete_days_before_leak*24*60/sim_step_minutes)
    historical_max_loss=init_big_value
    overshoot_h = 0
    overshoot_l = 0.2
    anomaly_counter = 0
    anomaly_timer=0
    anomaly_counter_threshold = 4
    anomaly_timer_threshold = 48  #improvement, before it was 16
    stats_sample_period = int(7*60/sim_step_minutes)
    j=-1
    leak_det1=np.zeros(len(timeStamp))
    current_min_loss=0
    df_min_loss = pd.DataFrame(columns = ['Min_loss'])
    df_max_loss = pd.DataFrame(columns = ['Max_loss'])
    num_guess_links = 10
    leak_loc_time_window = 48#improvement
    il_delay = 0
    localization_done = False
    guess = pd.Series(data = np.zeros(len(wn.pipe_name_list)), index=wn.pipe_name_list, dtype=np.float64)
    
    #Calibration of parameters for indirect pressure measurement, usin the LSM
    #Select junctions with tank demands and calibrates the parameters to infer WDS pressure
    tank_junctions_namelist  = df_results.loc[scNum, 'Tank Junctions Namelist'].split(', ');
    unmeasured_junctions_namelist = df_results.loc[scNum, 'Unmeasured Junctions Namelist'].split(', ');
    fraud_junctions_namelist = df_results.loc[scNum, 'Fraud Junctions Namelist'].split(', ');
    cal_parameters = pd.DataFrame(columns = ['p1', 'p2', 'p3'])
    inf_node_press = ODest_node_press.copy()
    tank_junctions_namelist_copy = tank_junctions_namelist.copy()
    for tank_junction_name in tank_junctions_namelist_copy:
        #if the junction has unmeasured demand or fraud, it's removed from the namelist of junctions with tanks
        if (tank_junction_name in unmeasured_junctions_namelist) or (tank_junction_name in fraud_junctions_namelist):
            tank_junctions_namelist.remove(tank_junction_name)
        else:
            #Defines initial time index for calibration and the number of samples
            init_time_index = 0
            calibration_sample_period = 96 # 4 days

            #Make sure that demands for calibration aren't zero and takes a certain number of samples
            array_with_zeros = np.zeros((calibration_sample_period,2), dtype = float)
            array_with_zeros[:,0] = np.arange(init_time_index, init_time_index+calibration_sample_period)
            array_with_zeros[:,1] = ODdem[tank_junction_name].iloc[init_time_index:init_time_index+calibration_sample_period].to_numpy()
            array_without_zeros = np.array([x for x in array_with_zeros if x.all() != 0])
            n_samples = array_without_zeros.shape[0]

            #Create_arrays to register sample demand and pressure values
            q = array_without_zeros[:,1]
            h1 = ODest_node_press[tank_junction_name].iloc[array_without_zeros[:,0]].to_numpy()

            qelev2 = q**2
            qelev1_85 = q**1.85
            qelev4 = q**4
            qelev3_85 = q**3.85
            qelev3_7 = q**3.7

            #Finds calibration parameters
            A = np.array([[n_samples, qelev2.sum(), qelev1_85.sum()], [qelev2.sum(), qelev4.sum(), qelev3_85.sum()], [qelev1_85.sum(), qelev3_85.sum(), qelev3_7.sum()]])
            b = np.array([h1.sum(), (qelev2*h1).sum(), (qelev1_85*h1).sum()])
            cal_parameters.loc[tank_junction_name] = np.linalg.solve(A,b)

            #Calculates infered pressures for customers with particular tanks, except fraudsters        
            inf_node_press[tank_junction_name].iloc[0:len(loss)] = cal_parameters['p1'].loc[tank_junction_name] + cal_parameters['p2'].loc[tank_junction_name] *ODdem[tank_junction_name].iloc[0:len(loss)]**2 + cal_parameters['p3'].loc[tank_junction_name]*ODdem[tank_junction_name].iloc[0:len(loss)]**1.85

    #Graph plots for Master's Thesis - Example of indirect pressure measurement, based on ESTIMATED PRESSURE measurements calibration
    if plot_graphs==True and scNum==17:
        fig = plt.figure(20)
        tank_junction_name = '131'
        ax = fig.add_subplot(3,1,1)
        ax.set_title("(a) - Customer Node 131 without Particular Tank, far from leakage")
        ax.set_ylabel("Pressure head (m)")
        xvals = np.arange(len((loss)))*0.5
        ax.plot(xvals, ODest_node_press[tank_junction_name].iloc[0:len(xvals)], color='red', label='Estimated')
        ax.plot(xvals, inf_node_press[tank_junction_name].iloc[0:len(loss)], color='black', label='Infered from demand', linestyle='dashed')
        ax.legend(loc='upper right')

        tank_junction_name = '171'
        ax = fig.add_subplot(3,1,2)
        ax.set_title("(b) - Customer Node 171 with Particular Tank, near leakage)")
        ax.set_ylabel("Pressure head (m)")
        ax.plot(xvals, ODest_node_press[tank_junction_name].iloc[0:len(xvals)], color='green', label='Estimated')
        ax.plot(xvals, inf_node_press[tank_junction_name].iloc[0:len(loss)], color='black', label='Infered from demand', linestyle='dashed')
        ax.legend(loc='upper right')

        tank_junction_name = '113'
        ax = fig.add_subplot(3,1,3)
        ax.set_title("(c) - Customer Node 113 with Particular Tank and Fraud)")
        ax.set_ylabel("Pressure head (m)")
        ax.plot(xvals, ODest_node_press[tank_junction_name].iloc[0:len(xvals)], color='blue', label='Estimated')
        ax.plot(xvals, inf_node_press[tank_junction_name].iloc[0:len(loss)], color='black', label='Infered from demand', linestyle='dashed')
        ax.set_xlabel("Time (hours)")
        ax.legend(loc='upper right')

        plt.show()

    if plot_graphs==True:
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)
        ax.set_ylabel("Water Loss(L/s)")
        xvals = np.arange(len(loss))*0.5
        ax.plot(xvals, loss, color='black', label='Measured \n  water Loss')
        ax.legend(loc='upper left')
        
        plt.show()

        detection_labels = pd.read_csv(DATASET_FOLDER_NAME +'/Detection Results/leak_det_labels_sc_'+str(scNum)+'.csv', index_col=[0])
        detection_labels.index = pd.to_datetime(detection_labels.index)

        for tank_junction_name in tank_junctions_namelist:
            fig = plt.figure(tank_junction_name)
            ax = fig.add_subplot(1,1,1)
            xvals = np.arange(len((loss)))*0.5
            ax.plot(xvals, ODest_node_press[tank_junction_name].iloc[0:len(xvals)], color='red', label='Estimated')
            ax.plot(xvals, inf_node_press[tank_junction_name].iloc[0:len(loss)], color='black', label='Infered from demand', linestyle='dashed')
            ax.legend(loc='upper right')

            plt.show()

    for i in range(0,len(timeStamp),1):
        #Calculates min and max loss in initial historic period and subsequently repeats after each stats_sample_period
        if i<initial_historical_period:
            if i==initial_historical_period-1:
                k = loss.iloc[0:i].idxmin()
                current_min_loss = loss.loc[k]
                df_min_loss.loc[k] = current_min_loss

                k = loss.iloc[0:i].idxmax()
                current_max_loss = loss.loc[k]
                df_max_loss.loc[k] = current_max_loss

                historical_min_loss = df_min_loss.min(axis=0).iloc[0]
                historical_min_loss_mean = df_min_loss.mean(axis=0).iloc[0]
                historical_max_loss = df_max_loss.max(axis=0).iloc[0] 

        else:
            j = (i-initial_historical_period-1)%stats_sample_period
            if j==stats_sample_period-1 and (leak_det1[i-j:i-1].sum())==0:
                k = loss.iloc[i-j-1:i].idxmin()
                current_min_loss = loss.loc[k]
                df_min_loss.loc[k] = current_min_loss

                k = loss.iloc[i-j-1:i].idxmax()
                current_max_loss = loss.loc[k]
                df_max_loss.loc[k] = current_max_loss

        #Get current demand loss
        current_loss = loss.iloc[i]

        #Detects leak
        if i>=initial_historical_period:
            #Leak Detection
            if leak_det1[i-1]==0 and current_loss > historical_max_loss + overshoot_h*(historical_max_loss-historical_min_loss):
                if anomaly_counter==0:
                    il = i
                anomaly_counter+=1

            #if anomaly_counter!=0:#improvement
            anomaly_timer+=1#improvement

            if leak_det1[i-1]==1:
                leak_det1[i]=1
            
            if current_loss < historical_min_loss_mean + overshoot_l*(historical_max_loss-historical_min_loss_mean):#improvement, anomaly_timer WAS used here before
                leak_det1[i]=0
                anomaly_counter=0
                
                historical_min_loss = df_min_loss.min(axis=0).iloc[0]
                historical_min_loss_mean = df_min_loss.mean(axis=0).iloc[0]
                historical_max_loss = df_max_loss.max(axis=0).iloc[0] 

                anomaly_timer=0

            if anomaly_counter>=anomaly_counter_threshold or anomaly_timer>=anomaly_timer_threshold: #improvement, anomaly_timer WASN'T used here before
                if anomaly_counter<anomaly_counter_threshold and il<i-anomaly_timer_threshold+1:
                    il = i - int(anomaly_timer_threshold*0.5)

                leak_det1[i]=1
                anomaly_counter=0

                #Discard min and max values eventually recorded during leak detection time
                dtm_list = df_min_loss.index.to_list()
                for dtm in dtm_list:
                    if dtm>timeStamp[i-anomaly_timer]:
                        df_min_loss.drop(index=dtm)
                
                dtm_list = df_max_loss.index.to_list()
                for dtm in dtm_list:
                    if dtm>timeStamp[i-anomaly_timer]:
                        df_max_loss.drop(index=dtm)

                anomaly_timer=0


            #Plot graph for leak detection understanding
            if plot_graphs==True and detection_labels['Label'].iloc[i]==1:
                fig = plt.figure(10)
                ax = fig.add_subplot(2,1,1)
                ax.set_title('(a) Scenario '+str(scNum)+': Loss demand')
                ax.set_ylabel("Water Loss(L/s)")
                xvals = np.arange(len(loss))*0.5
                ax.plot(xvals, loss, color='blue', label='Measured water Loss')
                ax.plot(xvals, loss*0 + 
                        historical_min_loss_mean + overshoot_l*(historical_max_loss-historical_min_loss_mean),
                        color='red', label='Leakage detection thresholds')
                ax.plot(xvals, loss*0 + 
                        historical_max_loss + overshoot_h*(historical_max_loss-historical_min_loss),
                        color='red')
                ax.plot(xvals, detection_labels['Label']*historical_max_loss*1.2,
                        color='black', label='True label(scaled)', linestyle='dotted')
                ax.plot(xvals, detection_labels['leak_det1']*historical_max_loss*1.2,
                        color='black', label='Detection Result(scaled)', linestyle='dashed')
                ax.legend(loc='upper left')

                # ax = fig.add_subplot(3,1,2)
                # ax.set_title('(b) Zoom at False Positive detection')
                # ax.set_ylabel("Water Loss(L/s) - Zoom")
                # xvals = np.arange(len(loss))*0.5
                # ax.plot(xvals, loss, color='blue', label='Measured water Loss')
                # ax.plot(xvals, loss*0 + 
                #         historical_min_loss_mean + overshoot_l*(historical_max_loss-historical_min_loss_mean),
                #         color='red', label='Leakage detection thresholds')
                # ax.plot(xvals, loss*0 + 
                #         historical_max_loss + overshoot_h*(historical_max_loss-historical_min_loss),
                #         color='red')
                # ax.plot(xvals, detection_labels['Label']*current_loss*1.2,
                #         color='black', label='True label(scaled)', linestyle='dotted')
                # ax.plot(xvals, detection_labels['leak_det1']*current_loss*1.2,
                #         color='black', label='Detection Result(scaled)', linestyle='dashed')

                ax = fig.add_subplot(2,1,2)
                ax.set_title('(b) Zoom at leakage period')
                ax.set_ylabel("Water Loss(L/s) - Zoom")
                ax.set_xlabel("Time (hours)")
                xvals = np.arange(len(loss))*0.5
                ax.plot(xvals, loss, color='blue', label='Measured water Loss')
                ax.plot(xvals, loss*0 + 
                        historical_min_loss_mean + overshoot_l*(historical_max_loss-historical_min_loss_mean),
                        color='red', label='Leakage detection thresholds')
                ax.plot(xvals, loss*0 + 
                        historical_max_loss + overshoot_h*(historical_max_loss-historical_min_loss),
                        color='red')
                ax.plot(xvals, detection_labels['Label']*historical_max_loss*1.2,
                        color='black', label='True label(scaled)', linestyle='dotted')
                ax.plot(xvals, detection_labels['leak_det1']*historical_max_loss*1.2,
                        color='black', label='Detection Result(scaled)', linestyle='dashed')

                plt.show()


        #Leak localization
        if leak_det1[i]==1 and localization_done == False:
            localization_done = True

            difpresinfer = inf_node_press.copy().iloc[0,:]
            difpresinfer = difpresinfer.loc[tank_junctions_namelist]

            for tank_junction_name in tank_junctions_namelist:
                iterador = 0
                while ODdem[tank_junction_name].iloc[il+iterador] == 0:
                    iterador+=1
                c1 = ODest_node_press[tank_junction_name].iloc[il+iterador]
                c2 = inf_node_press[tank_junction_name].iloc[il+iterador]
                difpresinfer.loc[tank_junction_name] = c1-c2

            il = il+il_delay
            il_window = round(leak_loc_time_window*60/sim_step_minutes)+1
               
            wn_test = wn

            for patt_name in wn_test.pattern_name_list:
                patt = wn_test.get_pattern(patt_name)
                if patt_name == '1':
                    patt.multipliers = patt.multipliers[0:il_window]
                else:
                    patt.multipliers = patt.multipliers[il:il+il_window]

            for ctrl_name in wn_test.control_name_list:
                wn_test.remove_control(ctrl_name)

            wn_test.get_link('335').initial_status=wntr.network.base.LinkStatus.Open
            wn_test.get_link('10').initial_status=wntr.network.base.LinkStatus.Open

            # Set time parameters
            ## Energy pattern remove
            wn_test.options.energy.global_pattern = '""'#(matheus)wn.energy.global_pattern = '""'
            # Set time parameters
            wn_test.options.time.duration = (il_window-1)*60*sim_step_minutes
            wn_test.options.time.hydraulic_timestep = 60*sim_step_minutes
            wn_test.options.time.quality_timestep = 0
            wn_test.options.time.report_timestep = 60*sim_step_minutes
            wn_test.options.time.pattern_timestep = 60*sim_step_minutes
            wn_test.options.quality.parameter = "None"
            wn_test.options.hydraulic.demand_model = 'DD'
            wn_test.reset_initial_values()

            g1 = pd.DataFrame(data = np.ones([len(wn.pipe_name_list),5])*np.inf, index = wn.pipe_name_list, columns= ['a', 'b', 'c', 'd', 'e'] )
            g2 = pd.DataFrame(data = np.ones([len(wn.pipe_name_list),len(tank_junctions_namelist)])*np.inf, index = wn.pipe_name_list, columns= tank_junctions_namelist)

            for pipe_index in range(len(wn.pipe_name_list)):
                pipe_name = wn.pipe_name_list[pipe_index]
                if pipe_name!='10' and pipe_name!='335' and pipe_name!='60':
                    wn_test2 = wntr.morph.split_pipe(wn_test,pipe_name, pipe_name+'test', 
                    'test_node', split_at_point = 0.5)
                    wn_test2.add_pattern('P_test', loss.values[il:il+il_window]-historical_min_loss)#comment to test with real leak demand instead of loss
                    # wn_test2.add_pattern('P_test', real_leak_demand0.iloc[:,0].values[il:il+il_window])#uncomment to test with real leak demand instead of loss
                    wn_test2.get_node('test_node').add_demand(0.001,'P_test')
                    del wn_test2.get_node('test_node').demand_timeseries_list[0]

                    sim = wntr.sim.EpanetSimulator(wn_test2)
                    results = sim.run_sim()
                    
                    # print("Pipe "+pipe_name+" simulation ok")
                    if ((all(results.node['pressure']> 0)) !=True)==True:
                        print("Negative pressures!")

                    flows = results.link['flowrate']
                    flows = flows[:len(timeStamp)]
                    flows = from_si(INP_UNITS, flows, HydParam.Flow)
                    flows = flows.round(results_decimal_digits)
                    flows.index = timeStamp[il:il+il_window]
                    pump10_flows = flows['10']
                    pump335_flows = flows['335']
                    tank1_flows = flows['40']
                    tank2_flows = flows['50']
                    tank3_flows = flows['20']

                    pres = results.node['pressure']
                    pres = pres[:len(timeStamp)]
                    pres = from_si(INP_UNITS, pres, HydParam.Pressure)
                    pres = pres.round(results_decimal_digits)
                    pres.index = timeStamp[il:il+il_window]
                    
                    g1.loc[pipe_name,'a']  = ((pump10_flows + ODdem['Lake'].iloc[il:il+il_window])**2).sum()
                    g1.loc[pipe_name,'b']  = ((pump335_flows + ODdem['River'].iloc[il:il+il_window])**2).sum()
                    g1.loc[pipe_name,'c']  = ((tank1_flows + ODdem['1'].iloc[il:il+il_window])**2).sum()
                    g1.loc[pipe_name,'d']  = ((tank2_flows + ODdem['2'].iloc[il:il+il_window])**2).sum()
                    g1.loc[pipe_name,'e']  = ((tank3_flows + ODdem['3'].iloc[il:il+il_window])**2).sum()
                   
                    for tank_junction_name in tank_junctions_namelist:
                        v1 = pres[tank_junction_name]
                        v2 = ODest_node_press[tank_junction_name].iloc[il:il+il_window]-difpresinfer.loc[tank_junction_name]
                        g2.loc[pipe_name,tank_junction_name]  = ((v2 - v1)**2).sum()

            guess = pd.Series(data = ((g1).product(axis=1)*(g1).sum(axis=1)).to_numpy(), index=wn.pipe_name_list, dtype=np.float64)
            guess = guess.sort_values(ascending=True)
            df_selected_pipes = guess.iloc[0:3*num_guess_links]
            df_selected_pipes = df_selected_pipes#*(g2).sum(axis=1)*(g2).product(axis=1)
            df_selected_pipes = df_selected_pipes.sort_values(ascending=True)

    print('Guess pipe list: \n'+str(df_selected_pipes.iloc[0:num_guess_links]))
    selected_pipes = df_selected_pipes.iloc[0:num_guess_links].index

    Labels = pd.read_csv(OD_folder_wntr +'/Labels.csv', index_col=[0])
    Labels.index = pd.to_datetime(Labels.index)

    detection_results_path_os = dataset_folder_os + '\\Detection Results'
    if not os.path.exists(detection_results_path_os):
        os.makedirs(detection_results_path_os)

    flabels = pd.DataFrame(leak_det1)
    flabels['Timestamp'] = timeStamp
    flabels = flabels.set_index(['Timestamp'])
    flabels.columns.values[0]='leak_det1'
    flabels['Label'] = Labels.loc[:,'Label']
    flabels.to_csv(detection_results_path_os+'\\leak_det_labels_sc_'+str(scNum)+'.csv')
    del flabels

    # Calculate accuracy
    accuracy = accuracy_score(Labels, leak_det1)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(Labels, leak_det1)
    print("Precision:", precision)

    # Calculate recall (sensitivity)
    recall = recall_score(Labels, leak_det1)
    print("Recall:", recall)

    # Calculate F1-score
    f1score = f1_score(Labels, leak_det1)
    print("F1-Score:", f1score)

    #SAVE RESULTS ARCHIVE
    iter = 0
    is_linked = 0
    is_in = 0
    true_leak_link = str(int(df_results.loc[scNum, 'Leak link 0 Name']))
    for iter in range(num_guess_links):
        if true_leak_link == selected_pipes[iter]:
            is_in=1

        if wn.get_link(true_leak_link).start_node == wn.get_link(selected_pipes[iter]).start_node or wn.get_link(true_leak_link).start_node == wn.get_link(selected_pipes[iter]).end_node or wn.get_link(true_leak_link).end_node == wn.get_link(selected_pipes[iter]).start_node or wn.get_link(true_leak_link).end_node == wn.get_link(selected_pipes[iter]).end_node:
            is_linked=1

        iter+=1

    is_exacly = 0
    if true_leak_link == selected_pipes[0]:
        is_exacly = 1

    df_results.loc[scNum, 'Detector 1 accuracy'] = accuracy
    df_results.loc[scNum, 'Detector 1 precision'] = precision
    df_results.loc[scNum, 'Detector 1 recall'] = recall
    df_results.loc[scNum, 'Detector 1 f1_score'] = f1score
    df_results.loc[scNum, 'Leaky pipe guess list'] = np.array2string(selected_pipes.values)  
    df_results.loc[scNum, 'True localization found?'] = is_exacly
    df_results.loc[scNum, 'True localization is within the list?'] = is_in
    df_results.loc[scNum, 'True localization is linked to pipe within the list?'] = is_linked
            
    return 1
    
if __name__ == '__main__':

    t = time.time()

    # runScenarios(17)

    NumScenarios = 501
    scArray = range(1, NumScenarios)
    
    for i in scArray:
        if df_results.loc[i, 'Number of Leaks']==1:
            try:
                runScenarios(i)
            except:
                pass

    fresults = dataset_folder_os+'\\Results.csv'
    df_results.to_csv(fresults)

    print('Total Elapsed time is '+str(time.time() - t) + ' seconds.')