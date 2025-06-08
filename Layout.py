# -*- coding: utf-8 -*-
"""
The code includes:
- A function to generate random layouts.
- A function to plot the layout.
- Two additional functions for wake visualization.

Authors
-------
Jacob Stentoft Broch
Signe Wisler Markussen

Last Updated
------------
Saturday, June 7, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon
import copy

# Floris import for visualization
from floris.flow_visualization import visualize_cut_plane
import floris.layout_visualization as layoutviz


def generate_random_layout(num_turb, boundaries, min_dist_D, D):
    """
    Positions turbines randomly within a rectangular domain while ensuring
    minimum spacing constraints.

    Parameters
    ----------
    num_turb : int
        Number of turbines for the generated layout.
    boundaries : list
        The boundaries of the domain (assumed rectangular).
    min_dist_D : float
        The minimum spacing requirement in terms of D.
    D : float
        The rotor diameter in meters.

    Returns
    -------
    layout_x: tuple
        X coordinates of placed turbines.
    layout_y: tuple
        Y coordinates of placed turbines.
    """
    # Extract domain size from boundaries (assuming rectangular domain)
    x_min, y_min = boundaries[0]
    x_max, y_max = boundaries[2]
    
    # Convert boundary to a polygon for spatial checks
    poly_outer = Polygon([boundaries[0], (x_max, y_min), boundaries[2], (x_min, y_max)])
    
    # Store valid positions
    positions = []
    min_distance = min_dist_D * D  # Minimum spacing in meters
    
    # Randomly choose the first turbine
    np.random.seed(42)  # Fixed seed for reproducibility
    init_x = np.random.uniform(x_min, x_max)
    init_y = np.random.uniform(y_min, y_max)
    
    while not poly_outer.contains(Point([init_x, init_y])):
        init_x = np.random.uniform(x_min, x_max)
        init_y = np.random.uniform(y_min, y_max)
    
    positions.append((init_x, init_y))
    
    # Iteratively place remaining turbines maximizing distance
    for _ in range(1, num_turb):
        max_dist = 0.0
        best_candidate = None
        
        for _ in range(1000):  # Limit attempts to find the best placement
            candidate_x = np.random.uniform(x_min, x_max)
            candidate_y = np.random.uniform(y_min, y_max)
            
            if not poly_outer.contains(Point([candidate_x, candidate_y])):
                continue
            
            dists = cdist([(candidate_x, candidate_y)], positions)
            min_dist = np.min(dists)
            
            if min_dist > max_dist and min_dist >= min_distance:
                max_dist = min_dist
                best_candidate = (candidate_x, candidate_y)
        
        if best_candidate is None:
            raise ValueError("Unable to place all turbines while maintaining the minimum distance.")
        
        positions.append(best_candidate)
    
    # Extract x and y coordinates
    layout_x, layout_y = zip(*positions)
    return layout_x, layout_y


def plot_layout(x_positions, y_positions, D, min_dist_D, L, Method, boundaries=None):
    """
    This function plot the wind turbines and the boundary.
    The axis of the plot are in terms of D (rotor diameter)

    Parameters
    ----------
    x_positions : tuple
        The x location of the turbines.
    y_positions : tuple
        The y location of the turbines.
    D : float64
        The rotor diameter in meters
    min_dist_D : int or float
        The minimum distance in terms of D.
    Method : str
        If Method = "Initial" plots the initial layout and boundary with blue squares for turbines.
        If Method = "Baseline" plots baseline layout and boundary and minimum distance circles with green stars for turbines.
        If Method = "Sequential" plots sequential layout and boundary and minimum distance circles with black circles for turbines.
    boundaries : list, optional
        The boundaries of the domain.
        If boundaries is input the plot also plots the boundary
        The default is None.

    Returns
    -------
    None.

    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    
    # Convert positions to rotor diameter scale
    x_positions = [x / D for x in x_positions]
    y_positions = [y / D for y in y_positions]

    # Plot boundaries if given
    if boundaries:
        boundary_x, boundary_y = zip(*boundaries)
        boundary_x = [x / D for x in boundary_x]  # Convert to D scale
        boundary_y = [y / D for y in boundary_y]  # Convert to D scale
        ax.plot(boundary_x, boundary_y, linestyle="--", color="b", linewidth=2)

    # Set marker and color depending on Method
    if Method == "Initial":
        marker, color, markersize = 's', 'blue', 6
        draw_circles = False
    elif Method == "Baseline":
        marker, color, markersize = '*', 'red', 10
        draw_circles = True
    elif Method == "Sequential":
        marker, color, markersize = 'o', 'black', 6
        draw_circles = True
    else:
        raise ValueError(f"Unknown Method: {Method}")
    
    ax.plot(x_positions, y_positions, marker=marker, linestyle="None", color=color, markersize=markersize)
    

    # Add circles around turbines for final layout
    if draw_circles:
        min_dist_radius = min_dist_D / 2  # Already in terms of D
        for x, y in zip(x_positions, y_positions):
            circle = plt.Circle((x, y), min_dist_radius, color='gray', fill=False, linestyle=":")
            ax.add_patch(circle)
    
    min_D_plot_margin = 3
  
    #ax.set_xlim(min(boundary_x) - min_dist_D,max(boundary_x) + min_dist_D)
    #ax.set_ylim(min(boundary_y) - min_dist_D,max(boundary_y) + min_dist_D)
    ax.set_xlim(-min_D_plot_margin,L + min_D_plot_margin)
    ax.set_ylim(-min_D_plot_margin,L + min_D_plot_margin)
    
    ticks = np.arange(0, L + 1, 5)      # [0, 5, 10, ... up to L]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t)}" for t in ticks])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t)}" for t in ticks])

    ax.tick_params(axis='both', labelsize=19)
    ax.set_xlabel("x/d", fontsize=19)
    ax.set_ylabel("y/d", fontsize=19)
    ax.grid(True)
    #if Method == "Baseline" or Method == "Sequential":
        #ax.set_title(f"{Method} Layout", fontsize=19)

    plt.tight_layout()
    plt.show()    


# Visualizing the wakes of the turbines at hub height for the top 4 power directions
def wake_visualization(fmodel, ind_speed, boundaries, row:int, opt_yaw_angles=None,max=False, Method = None):
    """
    This function plots a visualization of the wind farm using visualize_cut_plane and layoutviz.plot_turbine_rotors.
    It does this for either the wind directions that produced the maximum or minum amount of farm power.

    Parameters
    ----------
    fmodel : FLORIS model
        The FLORIS model containing the x and y locations and yaw angles of the turbines and the wind rose.
    ind_speed: float
        The index of the wind speeds that should be visualized.
    boundaries : list
         The boundaries of the domain (assumed rectangular).
    row : int
        The number of row and columns of the subplot.
        This means that the functions plots row^2 plots.
    opt_yaw_angles : Array of float64, optional
        Optional: Input of the optimized yaw angles. The default is None.
    max : bool, optional
        If max=True the function plots the wind directions that has the maximum farm power.
        If max=False the function plots the wind directions that has the minimum farm power. The default is False.
    Method : str
        Use "Initial", "Baseline" or "Sequential" for title based on which method it is.

    Returns
    -------
    None.

    """
    
    D = fmodel.core.farm.rotor_diameters[0][0] # rotor diameter
    
    fmodel.run()
    # Get farm power values
    farm_power = fmodel.get_farm_power()
    
    
    r = row**2
    if max:
        indices = np.argsort(farm_power[:, ind_speed])[-r:][::-1]  # Get top 4 highest power directions
        title_prefix = "Maximum"
    else:
        indices = np.argsort(farm_power[:, ind_speed])[:r]  # Get bottom 4 lowest power directions
        title_prefix = "Minimum"
    
    # List of wind directions with highest power
    ind_wind_directions = fmodel.wind_data.wind_directions[indices]
    wind_speed = fmodel.wind_data.wind_speeds[ind_speed]
    
    TI = fmodel.wind_data.ti_table[0,0]
    
    
    
    # Get power thrust table and number of turbines
    turbine_def = fmodel.core.farm.turbine_power_thrust_tables
    n_turbines = fmodel.n_turbines
    
    # Extract the power array (assuming there's only one turbine type)
    turbine_name = list(turbine_def.keys())[0]  # Get the turbine name
    
    # Get the wind speed and power where power is a function of wind speed
    turbine_power_curve_speeds = turbine_def[turbine_name]["wind_speed"]
    turbine_power_curve_power = turbine_def[turbine_name]["power"] * 1000  # Convert to W
        
    # Calculate total available farm power
    rated_power_per_turbine = np.max(turbine_power_curve_power)
    total_available_farm_power = n_turbines * rated_power_per_turbine
    
    # Calculate available farm power at the given wind speed
    power_at_wind_speed = np.interp(wind_speed, turbine_power_curve_speeds, turbine_power_curve_power)
    available_power_at_wind_speed = power_at_wind_speed * n_turbines
    
    

    fig, axes = plt.subplots(row, row, figsize=(15, 10))
    if Method in ["Initial", "Baseline", "Sequential"]:
        title_prefix = f"{Method} {title_prefix}"
    # Plot the turbine rotors
    if opt_yaw_angles is None:
        fig.suptitle(f"{title_prefix} Power Wind Flow Visualization at {wind_speed} m/s", fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f"{title_prefix} Power Wind Flow Visualization with Optimized Yaw Angles at {wind_speed} m/s", fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    plot_fmodel = copy.deepcopy(fmodel)
    
    # Loop through the top 4 wind directions and plot each one
    for i, (direction, power_rank) in enumerate(zip(ind_wind_directions, range(1, r + 1))):
        ax = axes[i]
        
        # Set wind data in FLORIS for the direction and speed
        plot_fmodel.set(wind_directions=[direction], wind_speeds=[wind_speed], turbulence_intensities=[TI])
        
        height = plot_fmodel.core.farm.hub_heights[0]  # Hub height
        
        L_x, L_y = map(max, zip(*boundaries))
        
        # Define desired square plot region in x-y space (fixed viewing area)
        bounds_margin_D = 10
        x_bounds = (-D*bounds_margin_D, L_x*D + D*bounds_margin_D)
        y_bounds = (-D*bounds_margin_D, L_y*D + D*bounds_margin_D)
        resolution_per_D = 12
        x_res = int((x_bounds[1] - x_bounds[0]) / D * resolution_per_D)
        y_res = int((y_bounds[1] - y_bounds[0]) / D * resolution_per_D)

        # Calculate the horizontal plane
        horizontal_plane = plot_fmodel.calculate_horizontal_plane(
            x_resolution=x_res,
            y_resolution=y_res,
            height=height,
            x_bounds=x_bounds,
            y_bounds=y_bounds
        )
        
        # Calculate normalized farm power
        normalized_rated_farm_power = farm_power[indices[i], ind_speed] / total_available_farm_power
        normalized_available_farm_power = farm_power[indices[i], ind_speed] / available_power_at_wind_speed
        
        # Plot the flow field with rotors
        visualize_cut_plane(
            horizontal_plane,
            ax=ax,
            label_contours=False,
            color_bar=True,
            title = (
                f"{power_rank}: Wind Flow at {direction}°\n"
                f"{normalized_rated_farm_power:.1%} of rated | "
                f"{normalized_available_farm_power:.1%} of available"
            ),
        )
        side_margin_D=4
        # Set fixed square view limits
        ax.set_xlim(-D*side_margin_D, L_x*D + D*side_margin_D)
        ax.set_ylim(-D*side_margin_D, L_y*D + D*side_margin_D)
        
        # Plot the turbine rotors
        if opt_yaw_angles is None:
            layoutviz.plot_turbine_rotors(plot_fmodel, ax=ax)
        else:
            layoutviz.plot_turbine_rotors(plot_fmodel, ax=ax, yaw_angles=opt_yaw_angles)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    




# Visualizing the wakes for a specific direction
def wake_visualization_Specific(fmodel, wind_speed, L_plot,wind_direction, opt_yaw_angles=None):
    """
    This function plots a visualization of the wind farm using visualize_cut_plane and layoutviz.plot_turbine_rotors.
    It does this for a specific wind direction.

    Parameters
    ----------
    fmodel : FLORIS model
        The FLORIS model containing the x and y locations and yaw angles of the turbines and the wind rose.
    windspped: float
        The index of the wind speeds that should be visualized.
    boundaries : list
         The boundaries of the domain (assumed rectangular).
    opt_yaw_angles : Array of float64, optional
        Optional: Input of the optimized yaw angles. The default is None.


    Returns
    -------
    None.

    """
    fmodel.run()
    D = fmodel.core.farm.rotor_diameters[0][0]
    TI = fmodel.wind_data.ti_table[0, 0]
    plot_fmodel = copy.deepcopy(fmodel)
    

    # Set wind conditions
    plot_fmodel.set(
       wind_directions=[wind_direction],
       wind_speeds=[wind_speed],
       turbulence_intensities=[TI],
       )
    if opt_yaw_angles is not None:
        plot_fmodel.set(yaw_angles=[opt_yaw_angles])

    height = plot_fmodel.core.farm.hub_heights[0]

    bounds_margin_D = 10
    x_bounds = (-D * bounds_margin_D, L_plot * D + D * bounds_margin_D)
    y_bounds = (-D * bounds_margin_D, L_plot * D + D * bounds_margin_D)
    resolution_per_D = 12
    x_res = int((x_bounds[1] - x_bounds[0]) / D * resolution_per_D)
    y_res = int((y_bounds[1] - y_bounds[0]) / D * resolution_per_D)
    
 
    horizontal_plane = plot_fmodel.calculate_horizontal_plane(
       x_resolution=x_res,
       y_resolution=y_res,
       height=height,
       x_bounds=x_bounds,
       y_bounds=y_bounds
   )
    

    fig, ax = plt.subplots(figsize=(8, 6))

    #levels = np.linspace(0, wind_speed, wind_speed+1)
    
    ax = visualize_cut_plane(
       horizontal_plane,
       ax=ax,
       label_contours=False,
       color_bar=False
       #title=f"Wake Visualization at {wind_direction}°, {wind_speed} m/s"
    )
    
    

    

    side_margin_D = 2
    ax.set_xlim(-D * side_margin_D, L_plot * D + D * side_margin_D)
    ax.set_ylim(-D * side_margin_D, L_plot * D + D * side_margin_D)

    layoutviz.plot_turbine_rotors(plot_fmodel, ax=ax, yaw_angles=opt_yaw_angles)
    
    tick_spacing_D = 5
    
    xticks_D = np.arange(0, ax.get_xlim()[1], D*tick_spacing_D)
    yticks_D = np.arange(0, ax.get_ylim()[1], D*tick_spacing_D)
    
    ax.set_xticks(xticks_D)
    ax.set_xticklabels([f"{int(x/D)}" for x in xticks_D])
    
    ax.set_yticks(yticks_D)
    ax.set_yticklabels([f"{int(y/D)}" for y in yticks_D])
    
    ax.set_xlabel("x/d", fontsize=19)
    ax.set_ylabel("y/d", fontsize=19)
    
    
    ax.tick_params(axis='both', which='major', labelsize=19)

    
    plt.tight_layout()
    plt.show()














