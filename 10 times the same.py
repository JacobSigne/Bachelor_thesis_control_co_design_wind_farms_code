# -*- coding: utf-8 -*-
"""
The code runs a single optimization case 10 times to perform a robustness analysis. 
It saves the results for both the Baseline and Sequential strategies for each run.

Authors
-------
Jacob Stentoft Broch
Signe Wisler Markussen

Last Updated
------------
Saturday, June 7, 2025
"""

# General import
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as timerpc # For timing the code
from datetime import datetime
import copy



# Floris Model import
from floris import FlorisModel, ParFlorisModel

# Import Wind Roses
import WindRoses

# Import functions related to layout, layout generation and layout visualization
import Layout


import ControlCodesign


# Import the context manager for saving the console and plot output.
# Remember to warp the main part of the code in: with save_output_context():
import Save_output
import os # Needed for saving variables in created folder
import pickle

   
# The main part of the code
if __name__ == '__main__':
    # Use the context manager to automatically save plots.
    # The entire code is wrapped with save_output_context() to get all plots.
    # Folder name is specified:
    folder_name = "10 times the same" + datetime.now().strftime("(%Y-%m-%d %H-%M-%S)")
    with Save_output.save_output(folder_name) as saved_folder:
        start_time = timerpc()
        #########################################
        # Setting up the FLORIS model, layout and Wind Rose
        
        # Set up FLORIS
        fmodel_1 = FlorisModel('inputs/gch_changed_turbulence_parameters.yaml')
        fmodel = ParFlorisModel(fmodel_1)
    
        # Define wind directions, wind speeds and turbulent intensity
        wind_directions = np.arange(0, 360, 10)
        wind_speeds = np.array([5.0, 7.0, 9.0, 11.0])  # Four wind speeds
        TI = 0.07 # Turbulence intensity
        
        # Calling the wind rose
        wind_rose = WindRoses.wind_rose_weibull(wind_directions, wind_speeds, TI)
    
        # Getting rotor diameter
        D = fmodel.core.farm.rotor_diameters[0] # rotor diameter
        
        # Parameters for result matrix
        min_dist_D = 4 # Minimum distance between turbines in terms of D
        max_min_D = 7
        aep_sacrifice = 0.0002 # AEP sacrifices
        
        # Number of turbines and boundary
        num_turb = 8 # Number of turbines
        k = 1.15
        L = k*np.sqrt(num_turb)*max_min_D # Boundary length in terms of number of D
        boundaries = [(0.0, 0.0), (0.0, L * D), (L * D, L * D), (L * D, 0.0), (0.0, 0.0)]
        
        # Generate random starting layout
        x_initial, y_initial = Layout.generate_random_layout(num_turb, boundaries, max_min_D, D)
        
        # Set wind data and layout in FLORIS
        fmodel.set(wind_data=wind_rose, layout_x=x_initial, layout_y=y_initial)
        
        # Plot initial positions
        Layout.plot_layout(x_initial, y_initial, D, max_min_D, L, Method = "Initial", boundaries=boundaries)
        
        
        # Array for saving gain
        gain_PD = np.zeros(10)
        gain_AEP = np.zeros(10)
        
        # Create a matrix (list of lists) to store all outputs
        all_results = [None] * len(gain_AEP)
        
        for i in range(10):
                call_fmodel = copy.deepcopy(fmodel)
                # Set wind data and layout in FLORIS
                call_fmodel.set(wind_data=wind_rose, layout_x=x_initial, layout_y=y_initial)
                (
                    BaselineModel,
                    baseline_boundaries,
                    codesign_aep,
                    baseline_aep,
                    codesign_power_density,
                    baseline_power_density,
                    baseline_time,
                    SEQModel,
                    sequential_boundaries,
                    sequential_aep,
                    sequential_power_density,
                    sequential_time,
                    seq_itertions_aep,
                    seq_iteration_pd
                    ) = ControlCodesign.Sequential(call_fmodel, boundaries, min_dist_D, aep_sacrifice)
                
                # Calculating gain from baseline to sequential 
                improvement_aep = 100*(sequential_aep - baseline_aep)/baseline_aep
                improvement_pd = 100*(sequential_power_density - baseline_power_density)/baseline_power_density
                
                # Store the result in the matrix
                gain_PD[i] = improvement_pd
                gain_AEP[i] = improvement_aep
                
                # Store all outputs in all_results
                baseline_layout_x = BaselineModel.layout_x
                baseline_layout_y = BaselineModel.layout_y
                baseline_yaw_angles = BaselineModel.core.farm.yaw_angles
                
                sequential_layout_x = SEQModel.layout_x
                sequential_layout_y = SEQModel.layout_y
                sequential_yaw_angles = SEQModel.core.farm.yaw_angles
                
                
                all_results[i] = (
                    baseline_layout_x,
                    baseline_layout_y,
                    baseline_yaw_angles,
                    baseline_boundaries,
                    codesign_aep,
                    baseline_aep,
                    codesign_power_density,
                    baseline_power_density,
                    baseline_time,
                    sequential_layout_x,
                    sequential_layout_y,
                    sequential_yaw_angles,
                    sequential_boundaries,
                    sequential_aep,
                    sequential_power_density,
                    sequential_time,
                    seq_itertions_aep,
                    seq_iteration_pd
                )
        
        # Save all_results using pickle
        filename = os.path.join(saved_folder, "results.pkl")
        
        to_save = {
            "all_results": all_results,
            "min_dist_D": min_dist_D,
            "aep_sacrifice": aep_sacrifice,
            "gain_pd": gain_PD,
            "gain_aep": gain_AEP,
            }
        
        with open(filename, "wb") as f:
            pickle.dump(to_save, f)
        
        # plot
        fig, ax = plt.subplots()
        ax.plot(gain_AEP, gain_PD,marker='o')
        
        ax.set_xlabel('Gain AEP [%]', size=12)
        ax.set_ylabel('Gain Power Density [%]', size=12)
    
        plt.show()
        


        
        

       