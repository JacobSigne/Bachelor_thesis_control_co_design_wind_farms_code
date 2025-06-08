# -*- coding: utf-8 -*-
"""
The code includes a function to generate wind roses using wind data 
from Horns Rev B and the Weibull distribution.

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
import matplotlib as mpl
from floris import WindRose

import scipy.special as spec

def wind_rose_weibull(wind_directions, wind_speeds, TI):
    """
    

    Parameters
    ----------
    wind_directions : numpy array
        Array of the wind directions.
    wind_speeds : numpy array
        Array of the wind speeds.
    TI : float
        The turbulent intentisy.

    Returns
    -------
    WindRose

    """
    def mph_to_mps(speed_mph):
        return speed_mph * 0.44704  # Conversion factor from mph to m/s
    # Define wind direction and speed intervals
    #wind_direction_intervals = [(i, i + 9) for i in range(0, 360, 10)]
    wind_direction_intervals = [(0, 4)] + [(i, i + 9) for i in range(5, 355, 10)] + [(355, 360)]
    #wind_speed_intervals = [(mph_to_mps(2.0), mph_to_mps(4.9)),
                             #(mph_to_mps(5.0), mph_to_mps(6.9)),
                             #(mph_to_mps(7.0), mph_to_mps(9.9)),
                             #(mph_to_mps(10.0), mph_to_mps(14.9)),
                             #(mph_to_mps(15.0), mph_to_mps(19.9)),
                             #(mph_to_mps(20.0), float('inf'))]
    wind_speed_intervals = [(2.0, 4.9),
                             (5.0, 6.9),
                             (7.0, 9.9),
                             (10.0,14.9),
                             (15.0, 19.9),
                             (20.0, float('inf'))]
    
    wind_speed_intervals = [(mph_to_mps(lo), mph_to_mps(hi)) for lo, hi in wind_speed_intervals]

    
    # Frequency data table provided
    # mph 2-4.9  5-6.9 7-9.9 10-14.9 15-19.9 20+  
    freq_data = [
        [0.143, 0.109, 0.152, 0.392, 0.096, 0.096], # 000-004
        [0.180, 0.124, 0.205, 0.299, 0.081, 0.034], # 005-014
        [0.137, 0.162, 0.156, 0.283, 0.156, 0.081], # 015-024
        [0.090, 0.100, 0.146, 0.352, 0.208, 0.342], # 025-034
        [0.087, 0.106, 0.137, 0.377, 0.311, 0.473], # 035-044
        [0.072, 0.115, 0.131, 0.377, 0.361, 0.744], # 045-054
        [0.093, 0.106, 0.137, 0.489, 0.348, 0.685], # 055-064
        [0.134, 0.134, 0.162, 0.504, 0.520, 0.759], # 065-074
        [0.096, 0.152, 0.180, 0.635, 0.442, 0.747], # 075-084
        [0.140, 0.143, 0.208, 0.613, 0.607, 0.986], # 085-094
        [0.103, 0.115, 0.215, 0.579, 0.548, 1.226], # 095-104
        [0.292, 0.261, 0.240, 0.737, 0.744, 1.817], # 105-114
        [0.252, 0.193, 0.212, 0.759, 0.663, 1.649], # 115-124
        [0.535, 0.398, 0.292, 0.672, 0.526, 1.335], # 125-134
        [0.190, 0.106, 0.196, 0.526, 0.445, 1.515], # 135-144
        [0.106, 0.109, 0.171, 0.538, 0.482, 1.161], # 145-154
        [0.124, 0.165, 0.171, 0.594, 0.445, 1.245], # 155-164
        [0.093, 0.096, 0.149, 0.548, 0.433, 1.186], # 165-174
        [0.115, 0.131, 0.177, 0.566, 0.411, 1.257], # 175-184
        [0.087, 0.128, 0.156, 0.495, 0.401, 1.326], # 185-194
        [0.230, 0.224, 0.380, 1.148, 0.709, 0.613], # 195-204
        [0.405, 0.314, 0.383, 0.663, 0.143, 0.012], # 205-214
        [0.538, 0.666, 0.772, 1.431, 0.473, 0.121], # 215-224
        [0.551, 0.694, 0.797, 1.223, 0.420, 0.202], # 225-234
        [0.361, 0.501, 0.507, 0.775, 0.177, 0.100], # 235-244
        [0.249, 0.386, 0.436, 0.809, 0.299, 0.467], # 245-254
        [0.190, 0.249, 0.333, 1.086, 0.688, 1.266], # 255-264
        [0.143, 0.221, 0.264, 0.899, 0.678, 1.777], # 265-274
        [0.180, 0.202, 0.221, 0.762, 0.597, 1.711], # 275-284
        [0.118, 0.137, 0.202, 0.601, 0.451, 0.983], # 285-294
        [0.205, 0.264, 0.395, 0.912, 0.436, 1.049], # 295-304
        [0.292, 0.417, 0.417, 0.909, 0.467, 0.467], # 305-314
        [0.859, 1.111, 1.204, 2.287, 0.840, 0.669], # 315-324
        [0.324, 0.439, 0.601, 1.080, 0.582, 0.722], # 325-334
        [0.162, 0.252, 0.255, 0.632, 0.305, 0.174], # 335-344
        [0.118, 0.168, 0.177, 0.433, 0.177, 0.134], # 345-354
        [0.143, 0.109, 0.152, 0.392, 0.096, 0.096], # 355-360
    ]
    
        
    # Convert frequency data into a NumPy array and normalize
    freq_table = np.array(freq_data)
    freq_table /= np.sum(freq_table)
    
    p,q = np.shape(freq_data)
        
    # Initialize a 2D array for wind conditions
    nDirections = len(wind_directions)
    nSpeeds = len(wind_speeds)
    wind_condition_freq = np.zeros((nDirections, q))

    
    # Loop through all directions and speeds using enumerate
    for direction_idx, direction in enumerate(wind_directions):
        dir_index = next((i for i, (low, high) in enumerate(wind_direction_intervals) if low <= direction <= high), None)
            
        if dir_index is not None:
            wind_condition_freq[direction_idx] = freq_table[dir_index]  

    wind_condition_freq = np.array(wind_condition_freq)    
    wind_condition_freq /= np.sum(wind_condition_freq)
    
    m,n = np.shape(wind_condition_freq)
    avg_freq = np.zeros(m)
    for i in range(m):
        avg_freq[i] = sum(wind_condition_freq[i])/100.

    speeds = np.array([2.+(4.9-2.)/2.,5.+(6.9-5.)/2.,7.+(9.9-7.)/2.,10.+(14.9-10.)/2.,15.+(19.9-15.)/2.,22.])
    #speeds = np.array([2.0, 4.9), (5.0, 6.9), (7.0, 9.9), (10.0,14.9), (15.0, 19.9), (20.0, float('inf'))]
    speeds = mph_to_mps(speeds)
    avg_speed = np.zeros(m)
    for i in range(m):
        for j in range(n):
            avg_speed[i] += wind_condition_freq[i][j]/100.*speeds[j]
        if avg_freq[i] > 0:
            avg_speed[i] /= avg_freq[i]
        else:
            avg_speed[i] = 0

        
    
    def Weibull(x,L):
        k = 2.0
        if L < 0.0001:
            L = 0.0001
        return (k/L)*(x/L)**(k-1)*np.exp(-(x/L)**k)
    
    
    freqs = np.zeros((nDirections, nSpeeds))
    for i in range(nDirections):
        dspeed = wind_speeds[1]-wind_speeds[0]
        num_int = 1000
        for j in range(nSpeeds):
            speed_int = np.linspace(wind_speeds[j]-dspeed/2.,wind_speeds[j]+dspeed/2.,num_int)

            k = 2.0
            scale = avg_speed[i]/(spec.gamma(1.0+1./k))

            freq_int = Weibull(speed_int,scale)
            speed_freq = np.trapz(freq_int,speed_int)
            #scale the weibull distribution by the direction's average frequency 
            #(amplitude change but shape is the same)
            #to match how often wind comes from that direction
            freqs[i,j] = speed_freq*avg_freq[i] 
    
    
    freqs /= np.sum(freqs)
    
    print("Wind speed bins:", wind_speeds)
    print("Summed frequencies per bin:", np.sum(freqs, axis=0))
    
    wind_rose = WindRose(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        freq_table=freqs,
        ti_table=TI
        )
    # Plot the windrose 
    ax = wind_rose.plot()
    fig = ax.figure
    
    # the plot is change so it also can show uneven numbers 
    
    # Remove the automatically added colorbar by checking all axes in the figure
    for axx in fig.axes:
        # If the axis has a ylabel matching "Wind speed [m/s]", remove it
        if hasattr(axx, 'get_ylabel') and axx.get_ylabel() == 'Wind speed [m/s]':
            fig.delaxes(axx)

    # Calculate the boundaries for the color mapping based on wind speed bins    
    step = np.diff(wind_speeds).mean()# Average difference between wind speed bins
    boundaries = np.concatenate([
            [wind_speeds[0] - step / 2], 
            wind_speeds[:-1] + step / 2, 
            [wind_speeds[-1] + step / 2]
            ])

    #create a color map and normalization for those bins
    cmap = plt.get_cmap("viridis_r", len(wind_speeds)) 
    norm = mpl.colors.BoundaryNorm(boundaries=boundaries, ncolors=len(wind_speeds))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    #Add a new colorbar with the correct ticks and labels
    cbar = fig.colorbar(sm, ax=ax, ticks=wind_speeds)
    cbar.set_ticklabels([str(ws) for ws in wind_speeds])
    cbar.set_label("Wind speed [m/s]")
    
    # Move colorbar a bit to the right
    pos = cbar.ax.get_position()  # get current position
    new_pos = [pos.x0 + 0.05, pos.y0, pos.width, pos.height]  # shift right by 0.05 (adjust as needed)
    cbar.ax.set_position(new_pos)

    plt.show()
    
    return wind_rose




