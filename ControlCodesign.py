# -*- coding: utf-8 -*-
"""
The code includes:
- A function that runs Co-design optimization.
- A function that runs the yaw optimization.
- A function that runs the Sequential control co-design strategy (zero-iteration is Baseline).
- A function that plots the AEP and PD for each Sequential iteration.
- A function that creates the result matrix.

Authors
-------
Jacob Stentoft Broch
Signe Wisler Markussen

Last Updated
------------
Saturday, June 7, 2025
"""


 
# General import
import matplotlib.pyplot as plt
from time import perf_counter as timerpc # For timing the code
import copy
import numpy as np

import Power

# Import our own Codesign Optimization Class
from codesign_optimization_random_search_squential_PD import CodesignOptimizationRandomSearch_Sequential


from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


def codesign(
        fmodel, 
        boundaries, 
        min_dist_D,
        aep_sacrifice,
        total_optimization_seconds,
        tolerance
        ):
    """
    This function runs the generic random search optimizion for the given input model and parameters.
    The function return the optimized locations of the wind turbines.

    Parameters
    ----------
    fmodel : FLORIS model
        The FLORIS model to optimize containing the inital x and y locations and the yaw angles 
        of the turbines and the wind rose.
    boundaries : list
        The boundaries of the domain.
    min_dist_D : int or float
        The minimum distance in terms of D.
    total_optimization_seconds : int or float
        The total number of second the optimizion method runs in each iteration.
    tolerance : float
        The percent tolerance for the codesign optimization loop.

    Returns
    -------
    x_opt : Array of float64
        The optimized x locations of the turbines.
    y_opt : Array of float64
        The optimized y locations of the turbines.

    """
    # Setting up the codesign optimizaation:    

    # Controls how far turbines move during random search:
    distance_pmf = None # None means it uses a uniform random distribution
    
    yaw_angles = fmodel.core.farm.yaw_angles # Get yaw angles from fmodel
    
    fmodel.run()
    aep_prev = fmodel.get_farm_AEP()
        
    power_density_prev = Power._get_power_density(fmodel, boundaries)
        
        
    # The model needs to be reset for the layout optimization can be run.
    # I don't know why but appetently it does.
    # This does not erase the layout or the wind rose
    fmodel.reset_operation()
    fmodel.set(yaw_angles=yaw_angles)
       
    boundaries_opt = boundaries
                
    i = 0
    stop_crit=20
        
    aep_improvement = tolerance + 1
    power_density_improvement = tolerance + 1
        
    while i < stop_crit and (aep_improvement > tolerance or power_density_improvement > tolerance):
        codesign_opt = CodesignOptimizationRandomSearch_Sequential(
            fmodel,
            boundaries_opt,
            min_dist_D=min_dist_D,
            seconds_per_iteration=10,
            total_optimization_seconds=total_optimization_seconds,
            distance_pmf=distance_pmf,
            use_dist_based_init=False, # Insures that the initial layout is the one we input
            aep_sacrifice = aep_sacrifice,
        )
            
        # Running the random search optimization
        aep_opt, power_density_opt, x_opt, y_opt, boundaries_opt, yaw_angles_opt = codesign_opt.optimize() 
            
        # Updating the layout after codesign optimization
        fmodel.set(layout_x=x_opt, layout_y=y_opt)
            
        # Calculating the AEP improvement and update previous AEP to current AEP
        aep_improvement = 100*(aep_opt - aep_prev)/aep_prev
        print(f"Iteration {i+1}: AEP Improvement = {aep_improvement:.7f} %")
        aep_prev = aep_opt
            
        # Calculating the Power Density improvement and update previous Power Density to current AEP
        power_density_improvement = 100 * (power_density_opt - power_density_prev) / power_density_prev
        print(f"Iteration {i+1}: Power Density Improvement = {power_density_improvement:.7f} %")
        power_density_prev = power_density_opt
            
        i+=1

    
    return x_opt, y_opt, boundaries_opt, aep_opt, power_density_opt, yaw_angles_opt


def yaw(fmodel, tolerance, min_yaw_angle=0, max_yaw_angle=25.0):
    """
    This function optimize the yaw angles of the FLORIS model using the Serial-Refine method.
    It runs the optimizer multiple times in a while loop until the change in the wind farm power is below a set tolerance.

    Parameters
    ----------
    fmodel : FLORIS model
        The FLORIS model to optimize containing the x and y locations and yaw angles of the turbines and the wind rose.
    min_yaw_angle : float, optional
        The minimum yaw angle.
        Default is 0 degrees
    max_yaw_angle : float, optional
        The maximum yaw angle.
        Default is 25 degrees
    tolerance : float
        The percent tolerance for the layout optimization loop.

    Returns
    -------
   opt_yaw_angles : Array of float64
       The optimized yaw angles.

    """ 
            
    yaw_angles_codesign = fmodel.core.farm.yaw_angles # Get yaw angles from fmodel
    
    yaw_first = YawOptimizationSR(
        fmodel,
        yaw_angles_baseline=yaw_angles_codesign,
        minimum_yaw_angle=min_yaw_angle,
        maximum_yaw_angle=max_yaw_angle
        )
    
    df_opt_first = yaw_first.optimize()
    
    # Compute total power improvement
    power_baseline = df_opt_first["farm_power_baseline"].sum()
    power_opt = df_opt_first["farm_power_opt"].sum()
    power_improvement = 100*(power_opt - power_baseline)/power_baseline
    
    # Run optimization iteratively and check stopping criterion
    if power_improvement < tolerance:
        return yaw_angles_codesign
    
    print(f"Iteration 1: Power Improvement = {power_improvement:.4f} %")
    
    optimized_yaw_angles = df_opt_first["yaw_angles_opt"].tolist()
        
    i = 1
    stop_crit=20
    
    
    
    while i < stop_crit and power_improvement > tolerance:
        # Initialize optimizer object
        yaw_opt = YawOptimizationSR(
            fmodel,
            yaw_angles_baseline=optimized_yaw_angles,
            minimum_yaw_angle=min_yaw_angle,
            maximum_yaw_angle=max_yaw_angle
            )
        df_opt = yaw_opt.optimize()
    
        # Compute total power improvement
        prev_power = df_opt["farm_power_baseline"].sum()
        power_opt = df_opt["farm_power_opt"].sum()
        power_improvement = 100*(power_opt - prev_power)/prev_power
        
        
    
        print(f"Iteration {i+1}: Power Improvement = {power_improvement:.7f} %")
        
        optimized_yaw_angles = df_opt["yaw_angles_opt"].tolist()
        
        i+=1
    
        
    opt_yaw_angles = np.vstack(df_opt["yaw_angles_opt"])    
    
    return opt_yaw_angles




################################################
# Sequiential control co-design framework
def Sequential(
        fmodel, 
        boundaries, 
        min_dist_D, 
        aep_sacrifice, 
        tolerance = 1e-5, 
        optimization_loop_seconds=3*60, 
        min_yaw_angle=0.0, 
        max_yaw_angle=25.0
        ):
    """
    

    Parameters
    ----------
    fmodel : FLORIS model
        The FLORIS model to optimize containing the x and y locations and yaw angles of the turbines and the wind rose.
    boundaries : list
        The boundaries of the domain.
    min_dist_D : int or float
        The minimum distance in terms of D.
    aep_sacrifice : float
        Permitted AEP sacrifice each time the layout optimization tries to shrink.
    tolerance : float, optional
        The tolerance used for the optimizations. The default is 1e-5.
    optimization_loop_seconds : int, optional
        The time in seconds that the co-design optimizations runs in each iteration. The default is 3*60 = 180 seconds.
    min_yaw_angle : float, optional
        The minimum yaw angles for the turbines in degrees. The default is 0.0.
    max_yaw_angle : float, optional
        The maximum yaw angles for the turbines in degrees. The default is 25.0.

    Returns
    -------
    BaselineModel : FLORIS model
        The FlorisModel for baseline.
    baseline_boundaries : list
        The boundaries for baseline.
    codesign_aep : float
        AEP after co-design optimization.
    baseline_aep : float
        AEP after baseline.
    baseline_power_density : float
        Power Density after baseline.
    baseline_time : float
        The time in seconds it took to run baseline.
    SEQModel : FLORIS model
        The FlorisModel for sequential.
    sequential_boundaries : list
        The boundaries for sequential.
    sequential_aep : float
        AEP after sequential.
    sequential_power_density : float
        Power Density after sequential.
    sequential_time : float
        The time in seconds it took to run baseline.
    seq_iteations_aep : numpy array
        Array that contains AEP for each sequential iteration.
    seq_iteration_pd : numpy array
        Array that contains Power Density for each sequential iteration.

    """
    
    # Calling wind data from fmodel (wind rose)
    wind_rose = fmodel.wind_data
    
    
    # Get initial values
    fmodel.run()
    aep_initial = fmodel.get_farm_AEP()
    aep_prev = aep_initial # The first previous AEP is set as the inital AEP.
    aep_improvement = tolerance + 1 # Initial AEP improvement is set to be larger than the tolerance for the loop to start.

    power_density_initial = Power._get_power_density(fmodel, boundaries)
    power_density_prev = power_density_initial # The first previous PD is set as the inital PD.
    power_density_improvement = tolerance + 1 # Initial PD improvement is set to larger than the tolerance for the loop to start.
    
    
    # Parameters for the sequential loop
    i = 0 # The iteration number of the loop
    max_iteration = 20 # The maximum number of iterations
  
    # The optimal boundary is set to be the initial boundary
    boundaries_opt = boundaries
    
    
    # Variables to save the found layouts, AEPs, PDs and yaw angles in.
    aep_codesign = []
    power_density_codesign = []
    
    power_density_seq_iteration = []
    aep_seq_iteration = []
    optimal_yaw_angles = []
    start_time = timerpc() # Start time for timing how long the sequential loop takes.
    print("=============================================================================================================")
    print("=============================================================================================================")
    print("=============================================================================================================")
    print(f"Sequential starts with Minimum Distance: {min_dist_D}, and AEP Sacrifice: {aep_sacrifice}")
    while i < max_iteration and (aep_improvement > tolerance or power_density_improvement > tolerance):
        print(f"-------------- Sequenital Iteration {i} starts --------------")
        # Running the co-design optimization and getting the optimized.
        x_opt, y_opt, boundaries_opt, aep, power_density, yaw_angles = codesign(
            fmodel,
            boundaries_opt,
            min_dist_D,
            aep_sacrifice,
            total_optimization_seconds = optimization_loop_seconds,
            tolerance=tolerance
            )
        
        
        # Updating the layout after codesign optimization
        fmodel.set(layout_x=x_opt, layout_y=y_opt, yaw_angles=yaw_angles) # Update model with new turbine positions
        
        aep_codesign.append(aep) # Save the AEP of the iteration
        power_density_codesign.append(power_density)
        
        
        # Running the yaw optimization
        opt_yaw_angles = yaw(
            fmodel,
            tolerance=tolerance,
            min_yaw_angle=min_yaw_angle,
            max_yaw_angle=max_yaw_angle
            )
        
        # Save the optimized yaw angles from the iteration
        optimal_yaw_angles.append(opt_yaw_angles) 
        
        # Updating the yaw_angles after yaw optimization
        fmodel.set(yaw_angles=opt_yaw_angles)
        
        # Getting AEP and Power Density after yaw optimization
        fmodel.run()
        aep = fmodel.get_farm_AEP()
        power_density = Power._get_power_density(fmodel, boundaries_opt)
        
        aep_seq_iteration.append(aep) # Save the AEP of the iteration
        power_density_seq_iteration.append(power_density) # Save the PD of the iteration
        
        # The model needs to be reset for the codesign optimization can be run again.
        # I don't know why but appetently it does.
        # This does not erase the layout or the wind rose.
        fmodel.reset_operation()
        fmodel.set(yaw_angles=opt_yaw_angles)
        
        if i < 1:
            # Saving Baseline values
            baseline_time = timerpc() - start_time
            
            baseline_x = x_opt
            baseline_y = y_opt
            baseline_yaw_angles = opt_yaw_angles
            baseline_boundaries = boundaries_opt
            codesign_aep = aep_codesign[0]
            codesign_power_density = power_density_codesign[0]
            baseline_aep = aep
            baseline_power_density = power_density
            

            # Calculates the improvement in AEP and PD from last iteration
            aep_improvement = 100*(baseline_aep - aep_initial)/aep_initial
            print(f"Sequential iteration {i}: AEP Improvement = {aep_improvement:.7f} %")
            power_density_improvement = 100*(power_density - power_density_prev)/power_density_prev
            print(f"Sequential iteration {i}: Power Density Improvement = {power_density_improvement:.7f} %")
            

            # Copying fmodel to create BaselineModel
            BaselineModel = copy.deepcopy(fmodel)
            BaselineModel.set(
                wind_data=wind_rose,
                layout_x=baseline_x,
                layout_y=baseline_y,
                yaw_angles=baseline_yaw_angles
                )
            
        else:
            # Calculates the improvement in AEP and PD from last iteration
            aep_improvement = 100*(aep - aep_prev)/aep_prev
            print(f"Sequential iteration {i}: AEP Improvement = {aep_improvement:.7f} %")
            power_density_improvement = 100*(power_density - power_density_prev)/power_density_prev
            print(f"Sequential iteration {i}: Power Density Improvement = {power_density_improvement:.7f} %")
        
        # Saves the curent AEP and Power Density as the previous for the next iteration
        aep_prev = aep 
        power_density_prev = power_density
                   
        i+=1
    
    # Saving Sequential values        
    sequential_time = timerpc() - start_time # Timing how long the sequential loop took.
    
    sequential_x = x_opt
    sequential_y = y_opt
    sequential_yaw_angles = opt_yaw_angles
    sequential_boundaries = boundaries_opt
    sequential_aep = aep_seq_iteration[-1]
    sequential_power_density = power_density_seq_iteration[-1]
    
    # Copying fmodel to create Sequential Model
    SEQModel = copy.deepcopy(fmodel)
    SEQModel.set(
        wind_data=wind_rose,
        layout_x=sequential_x,
        layout_y=sequential_y,
        yaw_angles=sequential_yaw_angles
        )
    
    
    print(f"Baseline took {baseline_time} seconds")
    print(f"Sequential took {sequential_time} seconds")
    
    # AEP
    print(f"Baseline AEP: {baseline_aep/1e9:.2f} [GWh]")
    print(f"Sequential AEP: {sequential_aep/1e9:.2f} [GWh]")
    
    # The gain of baseline from initial
    baseline_aep_gain = 100*(baseline_aep - aep_initial)/aep_initial
    print(f"Percent AEP again from Baseline: {baseline_aep_gain:.4f}%")
    
    # The gain of sequential from initial
    sequential_aep_gain = 100*(sequential_aep - aep_initial)/aep_initial # The total percent gain from optimization
    print(f"Percent AEP gain with Sequential: {sequential_aep_gain:.4f}%")
    
    # The gain of sequential from baseline
    seq_baseline_aep_gain = 100*(sequential_aep - baseline_aep)/baseline_aep
    print(f"Percent AEP gain from adding Sequential: {seq_baseline_aep_gain:.4f}%")
    
    
    # Power Density
    print(f"Baseline Power Density: {baseline_power_density:.2f} [W/m^2]")
    print(f"Sequential Power Density: {sequential_power_density:.2f} [W/m^2]")
    
    # The gain of baseline from initial
    baseline_power_density_gain = 100*(baseline_power_density - power_density_initial)/power_density_initial
    print(f"Percent Power Density again from Baseline: {baseline_power_density_gain:.4f}%")
     
    # The gain of sequential from initial       
    sequential_power_density_gain = 100*(sequential_power_density - power_density_initial)/power_density_initial # The total percent gain from optimization
    print(f"Percent Power Density gain with Sequential: {sequential_power_density_gain:.4f}%")
    
    # The gain of sequential from baseline
    seq_baseline_power_density_gain = 100*(sequential_power_density - baseline_power_density)/baseline_power_density
    print(f"Percent Power Density gain from adding Sequential: {seq_baseline_power_density_gain:.4f}%")
    
    seq_iteations_aep = aep_seq_iteration
    seq_iteration_pd = power_density_seq_iteration
    
    return (
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
        seq_iteations_aep,
        seq_iteration_pd
        )

def plot_seq_iterations(seq_iteations_objective, aep = False, pd = False):
    """
    Plot AEP or Power Density over sequential iterations. Iteration 0 is baseline

    Parameters
    ----------
    seq_iteations_objective : TYPE
        The values for each iteration.
    aep : TYPE, optional
        Set True if objective is AEP. The default is False.
    pd : TYPE, optional
        Set True if objective is Power Density. The default is False.

    Returns
    -------
    Plot.

    """
    fig, ax = plt.subplots()
    ax.plot(range(len(seq_iteations_objective)), seq_iteations_objective, color='b')
    ax.set_xlabel("Sequential iterations")
    if aep:
        ax.set_ylabel("AEP [GWh]")
        # Convert values from Wh to GWh
        ax.set_yticklabels([f"{y/1e9:.2f}" for y in ax.get_yticks()])
    elif pd:
        ax.set_ylabel("Power Density [W/m^2]")
    ax.grid(True)
    plt.show()


import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

def create_result_matrix(result_matrix,
                         min_D_list,
                         AEP_sac_list,
                         value,
                         categories=None,
                         category_styles=None):
    """
    

    Parameters
    ----------
    result_matrix : numpy.ndarray
        The result matrix for gain in power denisty or gain in AEP 
        as a function of minimum distance and permitted AEP sacrifice 
    min_D_list : numpy array
        A list that contain the minimums distanst (only number that shall be multipled with d)
    AEP_sac_list : numpy array
        A list that contain the Permitted AEP sacrifice (as fractions, e.g., 0.1 = 10%)
    value : str
        Name that desribe the number in the matrix: 
        - 'Gain: AEP [%]'
        - 'Gain: Power density[%]'

    Returns
    -------
    Plot the result matrix ,- with a colormap

    """

    
    
    # Create a heatmap using Seaborn/Matplotlib
    plt.figure(figsize=(8, 6))

    # Set a colormap
    colormap = 'bwr'
    
    # Normalize so that 0 is at the center of the colorbar
    abs_max = max(abs(result_matrix.min().min()), abs(result_matrix.max().max()))
    vmin, vmax = -abs_max, abs_max
    divnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    

    # Create the heatmap
    ax = sns.heatmap(result_matrix, 
                     xticklabels=[f"{x*100:.2f}" for x in AEP_sac_list], # Multiply by 100 to get in percent 
                     yticklabels=[f"{y}d" for y in min_D_list],
                     cmap=colormap,
                     norm=divnorm,
                     annot=True,  # Annotate each cell with the numeric value
                     fmt='.2f',   # Format for annotation
                     cbar_kws={'label': value},  # Colorbar label
                     linewidths=0.5,  # Line width between cells
                     linecolor='black',
                     annot_kws={'size': 20})

    

    # Set labels
    ax.set_xlabel("Permitted AEP sacrifice [%]",fontsize=19) 
    ax.set_ylabel("Minimum distance [m]",fontsize=19)


 
    # Move the x-axis tick labels to the top
    ax.xaxis.set_label_position('top')  # Moves the x-axis label to the top
    ax.xaxis.tick_top()  # Moves the x-axis tick labels to the top
    
    # Set the ront sizes for tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=19)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=19)

        
   # Set the font size for the colorbar label
    ax.collections[0].colorbar.set_label(value, fontsize=19)  # Set font size for colorbar label
    ax.collections[0].colorbar.ax.tick_params(labelsize=19)

    # Plot each category's markers in the top-right corner of each cell
    if categories:
        # offset positions: x toward right, y toward top
        x_off, y_off = 0.8, 0.2
        for cat, coords in categories.items():
            style = category_styles.get(cat, {})
            for (r, c) in coords:
                ax.scatter(c + x_off, r + y_off,
                           transform=ax.transData,
                           **style)
    
   # Display the plot
    plt.tight_layout()
    plt.show()
    
    
    

