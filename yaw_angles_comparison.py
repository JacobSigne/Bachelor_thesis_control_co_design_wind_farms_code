# -*- coding: utf-8 -*-
"""
The code includes several functions for plotting yaw angles in different ways.

Authors
-------
Jacob Stentoft Broch
Signe Wisler Markussen

Last Updated
------------
Saturday, June 7, 2025
"""

import matplotlib.pyplot as plt
import numpy as np


def yaw_angles_dif(fmodel_baseline, fmodel_SEQ):
    
 """
 
 Parameters
 ----------
 fmodel_baseline : FLORIS model
     The FLORIS model containing:
         The x and y locations of the turbines
         The wind rose.
         The yaw angles
         Have to be the floris model when baseline is done running 
    
 fmodel_SEQ : FLORIS model
     The FLORIS model containing:
         The x and y locations of the turbines
         The wind rose.
         The yaw angles
         Have to be the floris model when sequentail is done running 


 Returns
 -------
 Returns plot for each turbine. 
 That show the diffrens in yaw angles (SEQ - baseline) for each uniqe windspeed as function of direction. 
 """
 # plot yaw angles for each turbine over winddirection for the baseline and sequential 
 # Get yaw angles and wind directions
 yaw_angles_all_baseline = fmodel_baseline.core.farm.yaw_angles  
 yaw_angles_all_SEQ = fmodel_SEQ.core.farm.yaw_angles  
 wind_directions= fmodel_baseline.wind_data.wind_directions        
 uniqueSpeeds = fmodel_baseline.wind_data.wind_speeds     
 number_unique_speeds = uniqueSpeeds.size
 

 # Function to split the matrix into matrices with only one wind direction
 def split_into_x_matrices(A, x):
    matrices = []  # List to hold the resulting matrices
    
    # Iterate over x to create each matrix
    for i in range(x):
        matrix_rows = []  # List to hold row indices for the current matrix
        # For each matrix, add rows starting from i and progressing by x steps
        for j in range(i, A.shape[0], x):
            matrix_rows.append(j)
        # Append the matrix with selected rows to the result list
        matrices.append(A[matrix_rows, :])  # Select only those rows, keep all columns
    
    return matrices

 # Split the yaw angles matrix by wind speed
 matrices_yaw_angles_Windspeed_baseline = split_into_x_matrices(yaw_angles_all_baseline, number_unique_speeds)
 matrices_yaw_angles_Windspeed_baseline= np.array(matrices_yaw_angles_Windspeed_baseline)

 matrices_yaw_angles_Windspeed_SEQ = split_into_x_matrices(yaw_angles_all_SEQ, number_unique_speeds)
 matrices_yaw_angles_Windspeed_SEQ = np.array(matrices_yaw_angles_Windspeed_SEQ)
 
 # Make the numbers abs. 
 matrices_yaw_angles_Windspeed_baseline = np.abs( matrices_yaw_angles_Windspeed_baseline)
 matrices_yaw_angles_Windspeed_SEQ = np.abs( matrices_yaw_angles_Windspeed_SEQ)
 
 
 #The diffence in yaw angles 
 Yaw_marices_dif=matrices_yaw_angles_Windspeed_SEQ-matrices_yaw_angles_Windspeed_baseline
 
 # Loop through turbines
 for i in range(fmodel_baseline.n_turbines):
    plt.figure(figsize=(8, 5))  # Create a new figure for each turbine
    
    colors = plt.cm.tab10(np.linspace(0, 1, number_unique_speeds))  # Generate distinct colors
    # Loop through wind speeds and plot yaw angles for turbine i
    for j in range(number_unique_speeds):
        
        color = colors[j]  # Assign the same color for both plots of the same wind speed
             
        plt.plot(
            wind_directions,  # Wind direction (x-axis)
            Yaw_marices_dif[j,:,i],  # Yaw angles for turbine i at wind speed j
            marker='o', linestyle='None',color=color,markerfacecolor='none',label=f"Wind Speed {uniqueSpeeds[j]} m/s")
            
    # Labels and title
    plt.xlabel("Wind Direction [°]")
    plt.ylabel("Difference Yaw Angle [°]")
    plt.title(f" Difference Yaw Angle vs. Wind Direction (Turbine {i+1})")
    plt.grid(True)
    plt.legend()
    plt.show()  # Display the plot for the current turbine before moving to the next






def sum_yaw_angles(fmodel_baseline, fmodel_SEQ):
    """
     Parameters
     ----------
     fmodel_baseline : FLORIS model
        The FLORIS model containing:
         The x and y locations of the turbines
         The wind rose.
         The yaw angles
        Have to be the floris model when baseline is done runnig 
    
    fmodel_SEQ :FLORIS model
        The FLORIS model containing:
         The x and y locations of the turbines
         The wind rose.
         The yaw angles
        Have to be the floris model when sequentail is done runnig 


    Returns
    -------
    Returns one plot:  
        Y: Sum of yaw angles for all turbines in the farm for each uniqe windspeed for both SEQ and baseline
        x: Winddirections 
   """
    # # Plot The SUM of yaw angles for each direction and each windspeed
    # Get yaw angles and wind directions
    yaw_angles_all_baseline = fmodel_baseline.core.farm.yaw_angles  
    yaw_angles_all_SEQ = fmodel_SEQ.core.farm.yaw_angles  
    wind_directions= fmodel_baseline.wind_data.wind_directions        
    uniqueSpeeds = fmodel_baseline.wind_data.wind_speeds        
    number_unique_speeds = uniqueSpeeds.size
    

    # Function to split the matrix into matrices with only one wind direction
    def split_into_x_matrices(A, x):
       matrices = []  # List to hold the resulting matrices
       
       # Iterate over x to create each matrix
       for i in range(x):
           matrix_rows = []  # List to hold row indices for the current matrix
           # For each matrix, add rows starting from i and progressing by x steps
           for j in range(i, A.shape[0], x):
               matrix_rows.append(j)
           # Append the matrix with selected rows to the result list
           matrices.append(A[matrix_rows, :])  # Select only those rows, keep all columns
       
       return matrices

    # Split the yaw angles matrix by wind speed
    matrices_yaw_angles_Windspeed_baseline = split_into_x_matrices(yaw_angles_all_baseline, number_unique_speeds)
    matrices_yaw_angles_Windspeed_baseline = np.array(matrices_yaw_angles_Windspeed_baseline)

    matrices_yaw_angles_Windspeed_SEQ = split_into_x_matrices(yaw_angles_all_SEQ, number_unique_speeds)
    matrices_yaw_angles_Windspeed_SEQ = np.array(matrices_yaw_angles_Windspeed_SEQ)


   # make yaw angle number for comparison (SUM yaw-angles(abs) for all turbins for each direction)
   # make all the yaw angles absolut 
    Absolut_matrix_baseline = np.abs(matrices_yaw_angles_Windspeed_baseline)
    Absolut_matrix_SEQ = np.abs(matrices_yaw_angles_Windspeed_SEQ)
    
   # Find the sum of each row (The Sum of the turbines abs. yaw angles calculated for each direction for each wind speed)
    Yaw_angle_numbers_baseline = np.sum(Absolut_matrix_baseline,axis=2)
    Yaw_angle_numbers_SEQ = np.sum(Absolut_matrix_SEQ,axis=2)
  
   # Plot The SUM of abs. yaw angles for each direction and each windspeed 
    plt.figure(figsize=(10, 5))  # Define the size of the figure 
    plt.ylim(-10, 110)
    
    
    colors = ['red','blue','green','black']
    
    for i in range(Yaw_angle_numbers_baseline.shape[0]):  # Loop through rows 
      color = colors[i]  # Assign the same color for both plt.plot of the same wind speed
      plt.plot(
         wind_directions,
         Yaw_angle_numbers_baseline[i, :], # SUM yaw_anles baseline 
         marker='o',linestyle='None',color=color,markerfacecolor='none',markeredgewidth=1.5, markersize=9,label=f"Baseline-WS {uniqueSpeeds[i]} m/s")
     
      plt.plot(
         wind_directions,
         Yaw_angle_numbers_SEQ[i, :],# SUM yaw_anles sequential
         marker='x',linestyle='None',color=color,markeredgewidth=1.5, markersize=9,label=f"SEQ-WS {uniqueSpeeds[i]} m/s")
      
    
      # Labels and title
    plt.xlabel("Wind Direction [°]",fontsize=16)
    plt.ylabel("Yaw Angle Sum [°]",fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()  # Display the plot for the current turbine before moving to the next
    


def sum_yaw_angles_dif(fmodel_baseline, fmodel_SEQ,save_path=None):
    
    """
     Parameters
     ----------
     fmodel_baseline : FLORIS model
        The FLORIS model containing:
         The x and y locations of the turbines
         The wind rose.
         The yaw angles
        Have to be the floris model when baseline is done runnig 
    
    fmodel_SEQ :FLORIS model
        The FLORIS model containing:
         The x and y locations of the turbines
         The wind rose.
         The yaw angles
        Have to be the floris model when sequentail is done runnig 


    Returns
    -------
    Returns one plot:  
        Y: The diffrence insum of yaw angles (for all turbines in the farm) for each uniqe windspeed. (SEQ - Baseline)
        x: Winddirections 
   """
    # Plot The differnce of abs. yaw angles for each direction and each windspeed 
    # Get yaw angles and wind directions
    yaw_angles_all_baseline = fmodel_baseline.core.farm.yaw_angles  
    yaw_angles_all_SEQ = fmodel_SEQ.core.farm.yaw_angles  
    wind_directions = fmodel_baseline.wind_data.wind_directions        
    uniqueSpeeds = fmodel_baseline.wind_data.wind_speeds          
    number_unique_speeds = uniqueSpeeds.size
    

    # Function to split the matrix into matrices with only one wind direction
    def split_into_x_matrices(A, x):
       matrices = []  # List to hold the resulting matrices
       
       # Iterate over x to create each matrix
       for i in range(x):
           matrix_rows = []  # List to hold row indices for the current matrix
           # For each matrix, add rows starting from i and progressing by x steps
           for j in range(i, A.shape[0], x):
               matrix_rows.append(j)
           # Append the matrix with selected rows to the result list
           matrices.append(A[matrix_rows, :])  # Select only those rows, keep all columns
       
       return matrices

    # Split the yaw angles matrix by wind speed
    matrices_yaw_angles_Windspeed_baseline = split_into_x_matrices(yaw_angles_all_baseline, number_unique_speeds)
    matrices_yaw_angles_Windspeed_baseline= np.array(matrices_yaw_angles_Windspeed_baseline)

    matrices_yaw_angles_Windspeed_SEQ = split_into_x_matrices(yaw_angles_all_SEQ, number_unique_speeds)
    matrices_yaw_angles_Windspeed_SEQ= np.array(matrices_yaw_angles_Windspeed_SEQ)


   # make yaw angle number for comparison (SUM yaw-angles(abs) for all turbins for each direction)
   # make all the yaw angles absolut 
    Absolut_matrix_baseline = np.abs(matrices_yaw_angles_Windspeed_baseline)
    Absolut_matrix_SEQ = np.abs(matrices_yaw_angles_Windspeed_SEQ)
    
   # Find the sum of each row (The Sum of the turbines abs. yaw angles calculated for each direction for each wind speed)
    Yaw_angle_numbers_baseline=np.sum(Absolut_matrix_baseline,axis=2)
    Yaw_angle_numbers_SEQ=np.sum(Absolut_matrix_SEQ,axis=2)
    
   #Differnce 
    DIF_yaw_sum=Yaw_angle_numbers_SEQ-Yaw_angle_numbers_baseline
   

   # Plot The differnce of abs. yaw angles for each direction and each windspeed 
    plt.figure(figsize=(10, 5))  # Define the size of the figure 
    
    colors = ['red','blue','green','black']
    markers = ['D', 'x', '^', 'o']
    for i in range(DIF_yaw_sum.shape[0]):  # Loop through rows 
      
      plt.plot(
         wind_directions,
         DIF_yaw_sum[i, :], # SUM yaw_anles baseline 
         marker=markers[i],
         linestyle='None',
         markerfacecolor='none',
         label=f"WS {uniqueSpeeds[i]} m/s",color=colors[i],
         markeredgewidth=2,
         markersize=11)
      
     
   # Labels and title
    plt.xlabel("Wind Direction [°]",fontsize=16)
    plt.ylabel("Difference in Yaw Angle Sum [°]",fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    
    plt.grid(True)
    plt.legend(fontsize=15)
    
    if save_path:
       # Save with fixed DPI to get consistent pixel dimensions
       plt.savefig(save_path, dpi=100)  
    plt.show()  # Display the plot for the current turbine before moving to the next



