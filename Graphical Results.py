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

print(["Run input file: ", DATASET_FOLDER_NAME])

INP_UNITS = FlowUnits.LPS

dataset_folder_os = os.getcwd()+'\\'+DATASET_FOLDER_NAME

# RUN SCENARIOS
def runScenarios(scNum, resultsFileName):
    df_results = pd.read_csv('Results/'+resultsFileName+'.csv', header=0, index_col=0)

    if df_results.loc[scNum, 'True localization found?']==1:
        localization_result_string = 'True localization found.'
    else:
        if df_results.loc[scNum, 'True localization is within the list?']==1:
            localization_result_string = 'True localization is within the list.'
        else:
            if df_results.loc[scNum, 'True localization is linked to pipe within the list?']==1:
                localization_result_string = 'True localization is linked to pipe within the list.'
            else:
                localization_result_string = 'True localization is not even linked to any pipe within the list.'
    fraud_junctions_namelist  = df_results.loc[scNum, 'Fraud Junctions Namelist'].split(', ')
    unmeasured_junctions_namelist = df_results.loc[scNum, 'Unmeasured Junctions Namelist'].split(', ')
    leaky_pipes_guess_string = df_results.loc[scNum, 'Leaky pipe guess list']
    leaky_pipes_guess_string = leaky_pipes_guess_string.replace('[', '')
    leaky_pipes_guess_string = leaky_pipes_guess_string.replace(']', '')
    leaky_pipes_guess_string = leaky_pipes_guess_string.replace('\'', '')
    leaky_pipes_guess_list = leaky_pipes_guess_string.split(' ')
    truly_leaky_pipe = str(int(df_results.loc[scNum, 'Leak link 0 Name']))
    fraud_leak_node_series = pd.Series(data = np.zeros(len(fraud_junctions_namelist)+len(unmeasured_junctions_namelist)+1),index=['leak_node0']+fraud_junctions_namelist+unmeasured_junctions_namelist)
    fraud_leak_node_series.loc['leak_node0'] = 1
    leaky_pipes_guess_series = pd.Series(data = np.zeros(len(leaky_pipes_guess_list)),index=leaky_pipes_guess_list, dtype=int)

    n = 1
    for p in leaky_pipes_guess_list:
        leaky_pipes_guess_series.loc[p] = n
        n += 1
    
    if df_results.loc[scNum, 'True localization is within the list?']==1:
        leaky_pipes_guess_series.loc[truly_leaky_pipe+'l'] = leaky_pipes_guess_series.loc[truly_leaky_pipe]
        leaky_pipes_guess_list.append(truly_leaky_pipe+'l')
    
    #Define paths
    OD_folder_wntr = DATASET_FOLDER_NAME + '/Scenario-'+str(scNum)
    OD_inp_file_wntr = OD_folder_wntr + '/' + WDSName +'_Scenario-'+str(scNum) + '.inp'

    Dleak = df_results.loc[scNum, 'Leak link 0 Max. Demand(LPS)']
    Dfraud = df_results.loc[scNum, 'Max. Total Unknown/Fraud demand(LPS)']
    Dleak_Dfraud = round(Dleak/Dfraud,1)

    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    wn = wntr.network.WaterNetworkModel(OD_inp_file_wntr)

    ax = wntr.graphics.plot_network(wn, ax=axes)

    ax = wntr.graphics.plot_network(wn, link_attribute=leaky_pipes_guess_series, link_width=5, title='Algorithm ' + resultsFileName + ', Scenario '+str(scNum)+' (Dleak/Dfraud = '+str(Dleak_Dfraud)+'): '+localization_result_string, link_cmap = plt.cm.Greens_r, link_colorbar_label='Pipe Guess List', 
                                    node_attribute=fraud_leak_node_series, node_size=50, node_cmap = plt.cm.cool, node_colorbar_label='Fraud=0, Leak=1', ax=ax)
    
    fig_manager = plt.get_current_fig_manager()         
    fig_manager.window.state('zoomed')

    # plt.show()

    fig.savefig(str(scNum)+'.pdf')

    plt.close()

    return 1
    
if __name__ == '__main__':

    t = time.time()

    # runScenarios(4, 'I.a')

    NumScenarios = 501
    scArray = range(1, NumScenarios)

    resultsFileName = input('Type .csv archive name (I.a, I.b, II, III, IV): ')
    df_results = pd.read_csv('Results/'+resultsFileName+'.csv', header=0, index_col=0)

    for i in scArray:
        if df_results.loc[i, 'Number of Leaks']==1 and float(df_results.loc[i, 'Detector 1 f1_score'])>=0.99:
            try:
                runScenarios(i, resultsFileName)
            except:
                pass

    fresults = dataset_folder_os+'\\Results.csv'
    df_results.to_csv(fresults)

    print('Total Elapsed time is '+str(time.time() - t) + ' seconds.')