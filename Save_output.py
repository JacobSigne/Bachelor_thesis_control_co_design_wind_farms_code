# -*- coding: utf-8 -*-
"""
The code includes:
- Function to use for saving results.
- Multiple functions for reloading stored results.

Authors
-------
Jacob and Signe

Last Updated
------------
Saturday, June 7, 2025
"""

# Use a context manager to intercept and save plots:
import matplotlib.pyplot as plt
import os
from datetime import datetime
from contextlib import contextmanager
import sys # For console output saving

@contextmanager
def save_output(folder_name=None):
    """
    Calling this function to have the console output and plots save in a new folder.
    The function have folder name as input. It has a default folder name if no
    # folder name is specified.

    Remember to wrap the main function of the code in the following way if this function is called.
    In the code that calls this function:
        
    from save_output import save_output_context 
        with save_output_context():
            # The main part of the code

    Parameters
    ----------
    folder_name : str, optional
        The name of the created folder. The default is None.

    Yields
    ------
    folder_name : str
        yields the created folder.

    """
    if folder_name is None:
        folder_name = "Unspecified folder name " + datetime.now().strftime("(%Y-%m-%d %H-%M-%S)")
    # Creating a unique folder with the name format "Sequential run (date time)":
   
    os.makedirs(folder_name, exist_ok=True) 
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    plot_counter = {'count': 1}  # Use a mutable object to track count inside inner function
    
    # Redirect console output to a file:
    log_file = os.path.join(folder_name, 'output.log')
    with open(log_file, 'w') as f:
        sys.stdout = f
        sys.stderr = f

        original_show = plt.show

        def save_and_show(*args, **kwargs):
            plot_name = f"{plot_counter['count']}.pdf"
            plt.savefig(
                os.path.join(folder_name, plot_name),
                bbox_inches='tight',
                pad_inches=0,
                dpi=300
            )
            plot_counter['count'] += 1
            original_show(*args, **kwargs)

        plt.show = save_and_show

        try:
            yield folder_name
        finally:
            plt.show = original_show
            sys.stdout = original_stdout
            sys.stderr = original_stderr


import pickle
from collections import namedtuple

ResultEntryOld = namedtuple("ResultEntry", [
    "baseline_layout_x",
    "baseline_layout_y",
    "baseline_yaw_angles",
    "baseline_boundaries",
    "codesign_aep",
    "baseline_aep",
    "baseline_power_density",
    "baseline_time",
    "sequential_layout_x",
    "sequential_layout_y",
    "sequential_yaw_angles",
    "sequential_boundaries",
    "sequential_aep",
    "sequential_power_density",
    "sequential_time",
    "seq_iterations_aep",
    "seq_iteration_pd",
])

def load_results_old(saved_folder):
    """
   Load and unpack results from a pickle file stored in `saved_folder`.
   
   Example code of how to use function:
       import Save_output
       
       saved_folder = "Result Matrix First Run"

       all_results, min_dist_D, aep_sacrifice, result_matrix_aep, result_matrix_pd = Save_output.load_results(saved_folder)

       # Get the namedtuple at i=0, j=0
       model_0_0 = all_results[0][0]
       print("Baseline AEP:", model_0_0.baseline_aep)
       print("Sequential power density:", model_0_0.sequential_power_density)

       # And the other arrays:
       print("Minimum distances D:", min_dist_D)
       print("AEP sacrifice array shape:", aep_sacrifice)

   Parameters
   ----------
   saved_folder : str
       Path to the directory containing the file 'results.pkl'.

   Returns
   -------
   all_results : List[List[ResultEntry]]
       2D list of ResultEntry namedtuples for each (i, j) entry in the saved data.
   min_dist_D : numpy.ndarray
       Array of minimum distance values corresponding.
   aep_sacrifice : numpy.ndarray
       Array of AEP sacrifice values corresponding.
   result_matrix_aep : numpy.ndarray
       Array of AEP result matrix values corresponding.
   result_matrix_pd : numpy.ndarray
       Array of power density result matrix values corresponding.
   """
    filename = os.path.join(saved_folder, "results.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    raw_all = data["all_results"]

    # Rebuild methods_results as a 2D list of namedtuples
    unpacked_all = [
        [ResultEntryOld(*raw_all[i][j]) for j in range(len(raw_all[i]))]
        for i in range(len(raw_all))
    ]
    
    all_results = unpacked_all
    min_dist_D = data["min_dist_D"]
    aep_sacrifice = data["aep_sacrifice"]
    result_matrix_aep = data["result_matrix_aep"]
    result_matrix_pd = data["result_matrix_pd"]

    return all_results, min_dist_D, aep_sacrifice, result_matrix_aep, result_matrix_pd  




    
ResultEntry = namedtuple("ResultEntry", [
    "baseline_layout_x",
    "baseline_layout_y",
    "baseline_yaw_angles",
    "baseline_boundaries",
    "codesign_aep",
    "baseline_aep",
    "codesign_power_density",
    "baseline_power_density",
    "baseline_time",
    "sequential_layout_x",
    "sequential_layout_y",
    "sequential_yaw_angles",
    "sequential_boundaries",
    "sequential_aep",
    "sequential_power_density",
    "sequential_time",
    "seq_iterations_aep",
    "seq_iteration_pd",
])

def load_results(saved_folder):
    """
   Load and unpack results from a pickle file stored in `saved_folder`.
   
   Example code of how to use function:
       import Save_output
       
       saved_folder = "Result Matrix First Run"

       all_results, min_dist_D, aep_sacrifice, result_matrix_aep, result_matrix_pd = Save_output.load_results(saved_folder)

       # Get the namedtuple at i=0, j=0
       model_0_0 = all_results[0][0]
       print("Baseline AEP:", model_0_0.baseline_aep)
       print("Sequential power density:", model_0_0.sequential_power_density)

       # And the other arrays:
       print("Minimum distances D:", min_dist_D)
       print("AEP sacrifice array shape:", aep_sacrifice)

   Parameters
   ----------
   saved_folder : str
       Path to the directory containing the file 'results.pkl'.

   Returns
   -------
   all_results : List[List[ResultEntry]]
       2D list of ResultEntry namedtuples for each (i, j) entry in the saved data.
   min_dist_D : numpy.ndarray
       Array of minimum distance values corresponding.
   aep_sacrifice : numpy.ndarray
       Array of AEP sacrifice values corresponding.
   result_matrix_aep : numpy.ndarray
       Array of AEP result matrix values corresponding.
   result_matrix_pd : numpy.ndarray
       Array of power density result matrix values corresponding.
   """
    filename = os.path.join(saved_folder, "results.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    raw_all = data["all_results"]

    # Rebuild methods_results as a 2D list of namedtuples
    unpacked_all = [
        [ResultEntry(*raw_all[i][j]) for j in range(len(raw_all[i]))]
        for i in range(len(raw_all))
    ]
    
    all_results = unpacked_all
    min_dist_D = data["min_dist_D"]
    aep_sacrifice = data["aep_sacrifice"]
    result_matrix_aep = data["result_matrix_aep"]
    result_matrix_pd = data["result_matrix_pd"]

    return all_results, min_dist_D, aep_sacrifice, result_matrix_aep, result_matrix_pd  

def load_results_robust(saved_folder):
    """
   Load and unpack results from a pickle file stored in `saved_folder`.
   
   Example code of how to use function:
       import Save_output
       
       saved_folder = "Result Matrix First Run"

       all_results, min_dist_D, aep_sacrifice, result_matrix_aep, result_matrix_pd = Save_output.load_results(saved_folder)

       # Get the namedtuple at i=0, j=0
       model_0_0 = all_results[0][0]
       print("Baseline AEP:", model_0_0.baseline_aep)
       print("Sequential power density:", model_0_0.sequential_power_density)

       # And the other arrays:
       print("Minimum distances D:", min_dist_D)
       print("AEP sacrifice array shape:", aep_sacrifice)

   Parameters
   ----------
   saved_folder : str
       Path to the directory containing the file 'results.pkl'.

   Returns
   -------
   all_results : List[List[ResultEntry]]
       2D list of ResultEntry namedtuples for each (i, j) entry in the saved data.
   min_dist_D : numpy.ndarray
       Array of minimum distance values corresponding.
   aep_sacrifice : numpy.ndarray
       Array of AEP sacrifice values corresponding.
   result_matrix_aep : numpy.ndarray
       Array of AEP result matrix values corresponding.
   result_matrix_pd : numpy.ndarray
       Array of power density result matrix values corresponding.
   """
    filename = os.path.join(saved_folder, "results.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    raw_all = data["all_results"]

    # Rebuild methods_results as a 2D list of namedtuples
    unpacked_all = [ResultEntry(*item) for item in raw_all]
        
    all_results = unpacked_all
    min_dist_D = data["min_dist_D"]
    aep_sacrifice = data["aep_sacrifice"]
    gain_pd = data["gain_pd"]
    gain_aep = data["gain_aep"]

    return all_results, min_dist_D, aep_sacrifice, gain_pd, gain_aep  


def load_results_comp_times(saved_folder):
    """
   Load and unpack results from a pickle file stored in `saved_folder`.
   
   Example code of how to use function:
       import Save_output
       
       saved_folder = "Result Matrix First Run"

       all_results, min_dist_D, aep_sacrifice, result_matrix_aep, result_matrix_pd = Save_output.load_results(saved_folder)

       # Get the namedtuple at i=0, j=0
       model_0_0 = all_results[0][0]
       print("Baseline AEP:", model_0_0.baseline_aep)
       print("Sequential power density:", model_0_0.sequential_power_density)

       # And the other arrays:
       print("Minimum distances D:", min_dist_D)
       print("AEP sacrifice array shape:", aep_sacrifice)

   Parameters
   ----------
   saved_folder : str
       Path to the directory containing the file 'results.pkl'.

   Returns
   -------
   all_results : List[List[ResultEntry]]
       2D list of ResultEntry namedtuples for each (i, j) entry in the saved data.
   min_dist_D : numpy.ndarray
       Array of minimum distance values corresponding.
   aep_sacrifice : numpy.ndarray
       Array of AEP sacrifice values corresponding.
   result_matrix_aep : numpy.ndarray
       Array of AEP result matrix values corresponding.
   result_matrix_pd : numpy.ndarray
       Array of power density result matrix values corresponding.
   """
    filename = os.path.join(saved_folder, "results.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    raw_all = data["all_results"]

    # Rebuild methods_results as a 2D list of namedtuples
    unpacked_all = [ResultEntry(*item) for item in raw_all]
        
    all_results = unpacked_all
    min_dist_D = data["min_dist_D"]
    aep_sacrifice = data["aep_sacrifice"]
    num_turb = data["number_of_turbines"]
    baseline_comp_times = data["baseline_comp_times"]
    seq_comp_times = data["seq_comp_times"]

    return all_results, min_dist_D, aep_sacrifice, num_turb, baseline_comp_times, seq_comp_times  

def load_results_green_diamond(saved_folder):
    """
   Load and unpack results from a pickle file stored in `saved_folder`.
   
   Example code of how to use function:
       import Save_output
       
       saved_folder = "Result Matrix First Run"

       all_results, min_dist_D, aep_sacrifice, result_matrix_aep, result_matrix_pd = Save_output.load_results(saved_folder)

       # Get the namedtuple at i=0, j=0
       model_0_0 = all_results[0][0]
       print("Baseline AEP:", model_0_0.baseline_aep)
       print("Sequential power density:", model_0_0.sequential_power_density)

       # And the other arrays:
       print("Minimum distances D:", min_dist_D)
       print("AEP sacrifice array shape:", aep_sacrifice)

   Parameters
   ----------
   saved_folder : str
       Path to the directory containing the file 'results.pkl'.

   Returns
   -------
   all_results : List[List[ResultEntry]]
       2D list of ResultEntry namedtuples for each (i, j) entry in the saved data.
   min_dist_D : numpy.ndarray
       Array of minimum distance values corresponding.
   aep_sacrifice : numpy.ndarray
       Array of AEP sacrifice values corresponding.
   result_matrix_aep : numpy.ndarray
       Array of AEP result matrix values corresponding.
   result_matrix_pd : numpy.ndarray
       Array of power density result matrix values corresponding.
   """
    filename = os.path.join(saved_folder, "results.pkl")
    with open(filename, "rb") as f:
        data = pickle.load(f)

    raw_all = data["all_results"]

    # Rebuild methods_results as a 2D list of namedtuples
    unpacked_all = [ResultEntry(*item) for item in raw_all]
        
    all_results = unpacked_all
    min_dist_D = data["min_dist_D"]
    aep_sacrifice = data["aep_sacrifice"]

    return all_results, min_dist_D, aep_sacrifice 
  

