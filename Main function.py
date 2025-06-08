# -*- coding: utf-8 -*-
"""
Script to perform layout optimization of wind turbines using Baseline and Sequential strategies.

This script executes a total of 25 optimization cases by systematically varying two parameters:
1. The minimum allowed distance between turbines.
2. The acceptable level of AEP sacrifice.

Each case considers a layout of 8 turbines. Both the Baseline and Sequential optimization
strategies are applied to each case. The results are saved, and relevant plots are generated 
to visualize and compare the performance of both strategies across all cases.

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
from time import perf_counter as timerpc # For timing the code
from datetime import datetime
import copy



# Floris Model import
from floris import FlorisModel, ParFlorisModel

# Import Wind Roses
import WindRoses

# Import functions related to farm power
import Power

# Import functions related to layout, layout generation and layout visualization
import Layout

import yaw_angles_comparison

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
    folder_name = "Final results - 8 Turbines Last" + datetime.now().strftime("(%Y-%m-%d %H-%M-%S)")
    with Save_output.save_output(folder_name) as saved_folder:
        start_time = timerpc()
        #########################################
        # Setting up the FLORIS model, layout and Wind Rose
        
        # Set up FLORIS
        fmodel_base = FlorisModel('inputs/gch_changed_turbulence_parameters.yaml')
        fmodel = ParFlorisModel(fmodel_base)
    
        # Define wind directions, wind speeds and turbulent intensity
        wind_directions = np.arange(0, 360, 10)
        wind_speeds = np.array([5.0, 7.0, 9.0, 11.0])  # Four wind speeds
        TI = 0.07 # Turbulence intensity
        
        # Calling the wind rose
        wind_rose = WindRoses.wind_rose_weibull(wind_directions, wind_speeds, TI)
    
        # Getting rotor diameter
        D = fmodel.core.farm.rotor_diameters[0] # rotor diameter
        
        # Parameters for result matrix
        min_dist_D = np.arange(3, 8) # Minimum distance between turbines in terms of D
        aep_sacrifice = np.array([0, 0.0001, 0.0002, 0.0003, 0.0005]) # AEP sacrifices
        
        # Number of turbines and boundary
        num_turb = 8 # Number of turbines
        k = 1.15
        L = k*np.sqrt(num_turb)*max(min_dist_D) # Boundary length in terms of number of D
        boundaries = [(0.0, 0.0), (0.0, L * D), (L * D, L * D), (L * D, 0.0), (0.0, 0.0)]
        
        # Generate random starting layout
        x_initial, y_initial = Layout.generate_random_layout(num_turb, boundaries, max(min_dist_D), D)
        
        # Set wind data and layout in FLORIS
        fmodel.set(wind_data=wind_rose, layout_x=x_initial, layout_y=y_initial)
        
        # Plot initial positions
        Layout.plot_layout(x_initial, y_initial, D, max(min_dist_D), L, Method = "Initial", boundaries=boundaries)
        
        # Getting farm power as a function of wind direction for each wind speed.
        #PlotPower.farm_power_dic(fmodel, wind_rose, Method="Initial") # Farm power plot before layout optimization
        
        #row = 2 # Number of row and column to visualize
        #for ind_speed, vis_wind_speed in enumerate(wind_speeds):
            # row^2 plots for the max power directions
            #Layout.wake_visualization(fmodel, ind_speed, boundaries, row, max=False, Method="Initial")
            # row^2 plots for the min power directions
            #Layout.wake_visualization(fmodel, ind_speed, boundaries, row, max=True, Method="Initial")
        
        
        # Initialize empty result matricies
        result_matrix_aep = np.zeros((len(min_dist_D), len(aep_sacrifice)))
        result_matrix_pd = np.zeros((len(min_dist_D), len(aep_sacrifice)))
        
        # Array for for saving computational times
        baseline_comp_times = np.zeros((len(min_dist_D), len(aep_sacrifice)))
        seq_comp_times = np.zeros((len(min_dist_D), len(aep_sacrifice)))
        
        # Create a matrix (list of lists) to store all outputs
        all_results = [[None for _ in range(len(aep_sacrifice))] for _ in range(len(min_dist_D))]
        
        for i, min_D in enumerate(min_dist_D):
            for j, aep_sac in enumerate(aep_sacrifice):
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
                    seq_iteration_aep,
                    seq_iteration_pd
                    ) = ControlCodesign.Sequential(call_fmodel, boundaries, min_D, aep_sac, min_yaw_angle=0.0, max_yaw_angle=30.0)
                
                # Calculating gain from baseline to sequential for result matrix
                improvement_aep = 100*(sequential_aep - baseline_aep)/baseline_aep
                improvement_pd = 100*(sequential_power_density - baseline_power_density)/baseline_power_density
                
                # Store the result in the matrix
                result_matrix_aep[i, j] = improvement_aep
                result_matrix_pd[i, j] = improvement_pd
                
                # Store the computational times
                baseline_comp_times[i, j] = baseline_time
                seq_comp_times[i, j] = sequential_time
                
                # Store all outputs in all_results
                baseline_layout_x = BaselineModel.layout_x
                baseline_layout_y = BaselineModel.layout_y
                baseline_yaw_angles = BaselineModel.core.farm.yaw_angles
                
                sequential_layout_x = SEQModel.layout_x
                sequential_layout_y = SEQModel.layout_y
                sequential_yaw_angles = SEQModel.core.farm.yaw_angles
                
                
                all_results[i][j] = (
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
                    seq_iteration_aep,
                    seq_iteration_pd
                )
                
                ##################### BASELINE ######################
                
                # Plot final positions after Baseline with circles
                Layout.plot_layout(
                    baseline_layout_x, 
                    baseline_layout_y, 
                    D, 
                    min_D,
                    L,
                    Method="Baseline", 
                    boundaries=baseline_boundaries
                    )
                
                # Getting farm power as a function of wind direction for each wind speed.
                #PlotPower.farm_power_dic(BaselineModel,wind_rose, Method="Baseline") # Farm power plot before yaw optimization
                
                #baseline_yaw_angles = BaselineModel.core.farm.yaw_angles
                #for ind_speed, vis_wind_speed in enumerate(wind_speeds):
                    # row^2 plots for the max power directions
                    #Layout.wake_visualization(BaselineModel, ind_speed, baseline_boundaries, row, baseline_yaw_angles, max=False, Method="Baseline")
                    # row^2 plots for the min power directions
                    #Layout.wake_visualization(BaselineModel, ind_speed, baseline_boundaries, row, baseline_yaw_angles, max=True, Method="Baseline")
                
                
                #################### SEQUENTIAL ######################
                
                # Plot final positions after Sequential with circles
                Layout.plot_layout(
                    sequential_layout_x, 
                    sequential_layout_y, 
                    D, 
                    min_D,
                    L,
                    Method="Sequential",
                    boundaries=sequential_boundaries
                    )
                
                # Getting farm power as a function of wind direction for each wind speed.
                #PlotPower.farm_power_dic(SEQModel,wind_rose, Method="Sequential")
                                
                #sequential_yaw_angles = SEQModel.core.farm.yaw_angles
                # Visualizing the wakes of the turbines after optimization
                #for ind_speed, vis_wind_speed in enumerate(wind_speeds):
                    # row^2 plots for the max power directions
                    #Layout.wake_visualization(SEQModel, ind_speed, sequential_boundaries, row, sequential_yaw_angles, max=False, Method="Sequential")
                    # row^2 plots for the min power directions
                    #Layout.wake_visualization(SEQModel, ind_speed, sequential_boundaries, row, sequential_yaw_angles, max=True, Method="Sequential")
                               
                # Compare farm power between Baseline and Sequential
                Power.farm_power_comparison(BaselineModel, SEQModel)
                Power.farm_power_gain(BaselineModel, SEQModel)
                 
                # Compare yaw angles between Baseline and Sequential
                #yaw_angles_comparison.yaw_angles_dif(BaselineModel, SEQModel)
                yaw_angles_comparison.sum_yaw_angles(BaselineModel, SEQModel)
                yaw_angles_comparison.sum_yaw_angles_dif(BaselineModel, SEQModel)
                
                # Plot AEP and Power Density over sequential iterations. Iteration 0 is baseline
                ControlCodesign.plot_seq_iterations(seq_iteration_aep, aep=True)
                ControlCodesign.plot_seq_iterations(seq_iteration_pd, pd=True)
                        
        # Create result matrices
        ControlCodesign.create_result_matrix(result_matrix_aep, min_dist_D, aep_sacrifice, 'Gain: AEP [%]')
        ControlCodesign.create_result_matrix(result_matrix_pd, min_dist_D, aep_sacrifice, 'Gain: Power density[%]')
        
        # Total time
        total_time_sec = timerpc() - start_time
        
        # Convert seconds to HH:MM:SS format
        hours, remainder = divmod(total_time_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Print the result in both seconds and HH:MM:SS format
        print(f"The entire code took {total_time_sec} sec ({hours:02}:{minutes:02}:{seconds:02}).")
        
        # Save all_results using pickle
        filename = os.path.join(saved_folder, "results.pkl")
        
        to_save = {
            "all_results": all_results,
            "min_dist_D": min_dist_D,
            "aep_sacrifice": aep_sacrifice,
            "result_matrix_aep": result_matrix_aep,
            "result_matrix_pd": result_matrix_pd,
            }
        
        with open(filename, "wb") as f:
            pickle.dump(to_save, f)


        
        