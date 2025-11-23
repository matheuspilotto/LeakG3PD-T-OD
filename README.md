# LeakG3PD-OD: a Dataset for Water Distribution System Leakage Detection and Localization Using Water Demand Automated Meter Reading

This dataset was based on [LeakG3PD-T EPANET Net 3 dataset](https://github.com/matheuspilotto/LeakG3PD-T), which contains demand patterns modelled from customers with particular tanks, and was considered as representative of the Real System. 
The objective was to evaluate the use of almost exclusively demand signals for Leakage Detection and Localization.
In order to do that, similarly to the steps to generate [LeakG3PD-OD](https://github.com/matheuspilotto/LeakG3PD-OD):
- node demand values were copied (as if measured) from Real System and preprocessed including missing values, frauds and measurement errors
- in the original System Model (the one used as basis to create the 500 LeakG3PD-T Epanet Net 3 scenarios):
	- preprocessed demands were attributed to nodes
	- WDS tanks were replaced by reservoirs
	- pumps were replaced by negligible roughness pipes, NOT ADDING OPEN/CLOSE CONTROLS.
	- all reservoirs (including the tank substitutes) were attributed head patterns in order to reproduce pressures measured from Real System
	- original Real System pressures and link flows .csv files were deleted
	- simulation was run for all scenarios
	- System Model estimated pressures and estimated link flows .csv files were saved
	- once leak nodes don't exist in System Model, they were referred by their distance to leak link start nodes and such information was saved in "Leaks" folder
	- such dataset was named LeakG3PD-T-OD, where OD stands for Only Demands.

Using LeakG3PD-T-OD dataset, leak detection and localization algorithms similar, but improved in relation to the ones for LeakG3PD-OD dataset, were developed: 

<a href="https://drive.google.com/file/d/1spBIJG---tCUBPFKc2CphP4QUSf_cJfd/view?usp=sharing"><img src="https://drive.google.com/file/d/1spBIJG---tCUBPFKc2CphP4QUSf_cJfd" width="600" height="240"/><a>

The results showed that both algorithms are useful, but further developments are required for practical utilization. 
The dataset allows to calibrate infered pressures based on the demands of customers with particular tanks:

<a href="https://drive.google.com/file/d/1bEmpHzXEC1fjBBEqlCJ8vQCDNrP9nn_D/view?usp=sharing"><img src="https://drive.google.com/file/d/1bEmpHzXEC1fjBBEqlCJ8vQCDNrP9nn_D" width="600" height="240"/><a>

# Instructions for reproducibility

In any case, to run Python programs:
1. Download Anaconda3
2. Install VS Code
3. conda install conda-forge::wntr

If you wish to generate LeakG3PD-T-OD from scratch:
4. Download and unzip [LeakG3PD-T Epanet Net 3 dataset](https://drive.google.com/file/d/1lrjTWyReSL5X8qTa2dqtJ5Bm0GkbVLhe/view) folder into a "LeakG3PD-T-OD Evaluation" folder
5. Run python source file LeakG3PD-T-OD-DatasetGenerator.py
6. Rename the dataset folder as EPANET Net 3_T_OD

If you just wish to run the LDL program:
7. Download and unzip  [LeakG3PD-T-OD Epanet Net 3 dataset](xxxxxxxxx)
8. Run python source file T-OD-LeakDetectionAndLocalization.py
