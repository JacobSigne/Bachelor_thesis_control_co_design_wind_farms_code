
"""
The code includes a function to calculate power density, as well as several functions
to plot the wind farm's power output.

Authors
-------
Jacob Stentoft Broch
Signe Wisler Markussen

Last Updated
------------
Saturday, June 7, 2025
"""


import matplotlib.pyplot as plt


from shapely.geometry import Polygon, MultiPolygon

def _get_power_density(
        fmodel,
        poly_outer
        ):
    """
    Calculates the power density of the wind farm

    Parameters
    ----------
    fmodel : FLORIS model
        The FLORIS model to optimize containing the x and y locations and yaw angles of the turbines and the wind rose.
    poly_outer : list or MultiPolygon
        The boundary of the wind farm, either as a list or MultiPolygon.

    Raises
    ------
    TypeError
        Raises an error if poly_outer is neither list or MultiPolygon.

    Returns
    -------
    power_density : float
        Power density of the wind farm.

    """
    
    # Check the type of poly_outer
    if isinstance(poly_outer, MultiPolygon):
        pass  # Already correct type, do nothing
    elif isinstance(poly_outer, list):  # Assuming it's a list of boundary vertices
        poly_outer = MultiPolygon([Polygon(poly_outer)])
    else:
        raise TypeError("poly_outer must be either a MultiPolygon or a list of boundary vertices")
    fmodel.run()
    farm_power = fmodel.get_expected_farm_power()
    
    area = poly_outer.area
    power_density = farm_power/area
    return power_density

def farm_power_dic(fmodel, wind_rose, Method = None):
    """
    This function plots the farm power as a function of the wind direction given in the wind rose.
    It does this for every wind speed in the wind rose.

    Parameters
    ----------
    fmodel : FLORIS model
        The FLORIS model containing the x and y locations and yaw angles of the turbines and the wind rose.
    wind_rose : WindRose
        The wind rose containing the wind directions, wind speed, TI and frequency table.
    Method : str
        Use "Initial", "Baseline" or "Sequential" for title based on which method it is.

    Returns
    -------
    None.

    """

    # Plot the results
    fmodel.run()
    farm_power = fmodel.get_farm_power()
    # Plotting farm power as a function of wind speed and direction
    fig, ax = plt.subplots()
    for w_idx, wd in enumerate(wind_rose.wind_speeds):
        ax.plot(wind_rose.wind_directions, farm_power[:,w_idx] / 1e6, label=f"WD: {wd}")

    ax.set_xlabel("Wind Direction")
    ax.set_ylabel("Power [MW]")
    ax.grid(True)
    ax.legend()
    if Method in ["Initial", "Baseline", "Sequential"]:
        ax.set_title(Method + " Farm Power" )
    else:
        ax.set_title("Farm Power" )
    plt.show()
    
    
    
def farm_power_comparison(fmodel_baseline, fmodel_SEQ):
    """
    Compares the farm power of two models 
    The farm power for both models over directions. 
    
    
    Input Parameters:
    -----------------
    fmodel_baseline : FLORIS model
        The FLORIS model containing:
            The x and y locations of the turbines
            The wind rose.
            The yaw angles
    
    fmodel_SEQ : FLORIS model
        The FLORIS model containing:
            The x and y locations of the turbines
            The wind rose.
            The yaw angles
        
    Output:
    --------------
    SubPlot for the wind speeds in the wind rose showing farmPower for both models as a function of windDirection.
    
    """
    
    # Run both models
    fmodel_baseline.run()
    fmodel_SEQ.run()
    
    wind_rose = fmodel_baseline.wind_data
    
    # Get farm power data
    farm_power_baseline = fmodel_baseline.get_farm_power()
    farm_power_SEQ = fmodel_SEQ.get_farm_power()
    
    # Create subplots for different wind speeds
    unique_speeds = wind_rose.wind_speeds  # The wind speeds available in wind_rose

    # Create subplots dynamically based on the number of wind speeds
    fig, axarr = plt.subplots(
        nrows=len(unique_speeds),  # One row per unique wind speed
        ncols=1,
        sharex=True,
        figsize=(10, 8))

        # If there is only one wind speed, axarr will not be an array, so make sure it can still be indexed
    if len(unique_speeds) == 1:
           axarr = [axarr]

        # Loop through the wind speeds and create the plot for each
    for ii, ws in enumerate(unique_speeds):
            ax = axarr[ii]  # Select the appropriate subplot for the wind speed

            # Extract the power for the current wind speed and plot
            power_baseline = farm_power_baseline[:, ii]  # Farm power for baseline model (column index by wind speed)
            power_SEQ = farm_power_SEQ[:, ii]  # Farm power for optimized model (column index by wind speed)

            # Plot baseline and optimized farm power
            ax.plot(wind_rose.wind_directions, power_baseline / 1e6, color='r', label='Baseline',linewidth=2.7)  # baseline (in MW)
            ax.plot(wind_rose.wind_directions, power_SEQ / 1e6, color='k', label='Sequential',linewidth=2.7)  # sequential (in MW)

            # Move title into the plot (bottom-left corner)
            ax.text(0.02, 0.05, f"Wind Speed = {ws:.0f} m/s", transform=ax.transAxes, fontsize=17,
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            ax.tick_params(axis='both', labelsize=16)

            ax.grid(True)
            #ax.set_ylabel('Farm Power [MW]', fontsize=15)

            if ii == len(unique_speeds) - 1:
                ax.set_xlabel('Wind Direction [°]', fontsize=17)

            if ii == 0:
                ax.legend(loc='upper right',fontsize=17)

    # Adjust layout to prevent overlap
    fig.align_ylabels(axarr)
    fig.text(0.04, 0.5, 'Farm Power [MW]', va='center', rotation='vertical', fontsize=17)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.1)
    plt.savefig("farm_power_plot.pdf", dpi=300, bbox_inches='tight') 
    plt.show()


def farm_power_comparison_2(fmodel_co_design, fmodel_baseline, fmodel_SEQ):
    """
    Compares the farm power of three models.
    
    Input Parameters:
    -----------------
    fmodel_co_design    : FLORIS model
    fmodel_baseline : FLORIS model
    fmodel_SEQ      : FLORIS model
    
    
    Output:
    --------------
    Subplot of farm power for different wind speeds and directions for all three models.
    """
    
    # Run all models
    fmodel_co_design.run()
    fmodel_baseline.run()
    fmodel_SEQ.run()
    
    
    wind_rose = fmodel_baseline.wind_data
    
    # Get farm power data
    farm_power_baseline = fmodel_baseline.get_farm_power()
    farm_power_SEQ = fmodel_SEQ.get_farm_power()
    farm_power_co_design = fmodel_co_design.get_farm_power()
    
    # Wind speeds
    unique_speeds = wind_rose.wind_speeds
    
    custom_ylim_dict = {
    5: (2, 3.5), # have to be changed to match data 
    7: (6, 9.5), # have to be changed to match data
    9: (14, 20),# have to be changed to match data
    11: (24, 37),# have to be changed to match data
}

    # Create subplots dynamically
    fig, axarr = plt.subplots(
        nrows=len(unique_speeds),
        ncols=1,
        sharex=True,
        figsize=(10, 8)
    )

    if len(unique_speeds) == 1:
        axarr = [axarr]

    for ii, ws in enumerate(reversed(unique_speeds)):
        ax = axarr[ii]
        
        power_co_design = farm_power_co_design[:, len(unique_speeds) - 1 - ii]
        power_baseline = farm_power_baseline[:, len(unique_speeds) - 1 - ii]
        power_SEQ = farm_power_SEQ[:, len(unique_speeds) - 1 - ii]
        

        # Plot all three models
        ax.plot(wind_rose.wind_directions, power_co_design / 1e6, color='#1b9e77', label='Co-design', linewidth=2.7)
        ax.plot(wind_rose.wind_directions, power_baseline / 1e6, color='r', label='Baseline', linewidth=2.7)
        ax.plot(wind_rose.wind_directions, power_SEQ / 1e6, color='k', label='Sequential', linewidth=2.7)
       

        ax.text(0.02, 0.05, f"Wind Speed = {ws:.0f} m/s", transform=ax.transAxes, fontsize=17,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True)
        
        if ws in custom_ylim_dict:
            ax.set_ylim(custom_ylim_dict[ws])
            if ws == 7:
                ax.set_yticks([6, 7, 8, 9])  # exact ticks you want for WS=7
                # else: no custom ticks -> matplotlib chooses automatically

        if ii == len(unique_speeds) - 1:
            ax.set_xlabel('Wind Direction [°]', fontsize=17)

        if ii == 0:
            ax.legend(loc='upper right', fontsize=17)

    fig.align_ylabels(axarr)
    fig.text(0.04, 0.5, 'Farm Power [MW]', va='center', rotation='vertical', fontsize=17)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.1)
    plt.savefig("farm_power_plot.pdf", dpi=300, bbox_inches='tight') 
    plt.show()
    


def farm_power_gain(fmodel_baseline, fmodel_SEQ):
    """
    Compares the farm power of two models
    Percentage gain in farm power over directions.
    
    Input Parameters:
    -----------------
    fmodel_baseline : FLORIS model
        The FLORIS model containing:
            The x and y locations of the turbines
            The wind rose.
            The yaw angles
    
    fmodel_opt : FLORIS model
        The FLORIS model containing:
            The x and y locations of the turbines
            The wind rose.
            The yaw angles
    
    Output:
    --------------

    Plot the gain in farmPower as a function of windDirection for the wind speeds in the wind rose.
    """
    
    # Run both models
    fmodel_baseline.run()
    fmodel_SEQ.run()
    
    wind_rose = fmodel_baseline.wind_data
    
    # Get farm power data
    farm_power_baseline = fmodel_baseline.get_farm_power()
    farm_power_SEQ = fmodel_SEQ.get_farm_power()
  
    # Calculate percentage gain
    gain_farm_power = (farm_power_SEQ - farm_power_baseline) / farm_power_baseline * 100
    
    colors = ['red','blue','green','black']
        
    # Plot the percentage gain as function of directions
    plt.figure(figsize=(8, 6))
    for w_idx, ws in enumerate(wind_rose.wind_speeds):
        plt.plot(wind_rose.wind_directions, gain_farm_power[:, w_idx], label=f"Wind speed = {ws:.1f} m/s", color=colors[w_idx])
        
    plt.ylabel('Gain in Farm Power [%]', size=12)
    plt.xlabel('Wind Direction [°]', size=12)
    plt.legend()
    plt.grid(True)
    plt.title('Percentage Gain in Farm Power')
    plt.show()
