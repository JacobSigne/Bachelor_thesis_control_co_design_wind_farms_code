# -*- coding: utf-8 -*-
"""
This code is based on the FLORIS script titled "layout_optimization_random_search."  
Modifications have been made to include power density as an optimization objective  
and to treat the boundary as a design variable.  
Turbine control has been excluded from the optimization process.
This code corresponds to one outer iteration of the Co-design. 

Authors
-------
Jacob Stentoft Broch  
Signe Wisler Markussen

Last Updated
------------
Saturday, June 7, 2025
"""

from multiprocessing import Pool
from time import perf_counter as timerpc

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
from shapely.geometry import Point, Polygon, MultiPolygon

from floris import FlorisModel
from floris.optimization.yaw_optimization.yaw_optimizer_geometric import (
    YawOptimizationGeometric,
)

#from .layout_optimization_base import LayoutOptimization
from floris.optimization.layout_optimization.layout_optimization_base import LayoutOptimization

def _shrink_boundary_and_turbines(
        layout_x,
        layout_y,
        fmodel,
        poly_outer,
        min_dist,
        shrink_factor,
        yaw_angles,
        use_value,
        current_objective,
        current_power_density,
        aep_sacrifice
        ):
    """
    Shrinks the wind farm boundary and moves turbines closer to the centroid.
    Tries shrinking x and y directions independently.
    Ensures that turbines still respect the minimum distance requirement.
    
    Args:
        layout_x (list): X-coordinates of turbines.
        layout_y (list): Y-coordinates of turbines.
        poly_outer (MultiPolygon): Current wind farm boundary.
        shrink_factor (float): Fraction to shrink boundary per step.
        min_dist (float): Minimum turbine spacing.
    
    Returns:
        tuple: (Updated layout_x, Updated layout_y, Updated poly_outer)
               If shrinkage is not possible, returns the original values.
    """
    shrink_success = False
    
    current_layout_x = layout_x
    current_layout_y = layout_y
    current_poly_outer = poly_outer
       
    # Compute farm centroid
    farm_centroid = poly_outer.centroid
    
    # Try shrinking boundary in x direction
    x_polygons = []
    for poly in current_poly_outer.geoms:
        x_coords = [
            (
                farm_centroid.x + (x - farm_centroid.x) * shrink_factor,
                y
            )
            for x, y in poly.exterior.coords
        ]
        x_polygons.append(Polygon(x_coords))
    
    x_poly_outer = MultiPolygon(x_polygons)
    
    # Move turbines toward the centroid in x direction
    x_layout_x = [farm_centroid.x + (x - farm_centroid.x) * shrink_factor for x in current_layout_x]
    x_layout_y = current_layout_y
    
    # Check if new layout still meets minimum distance constraint
    if test_min_dist(x_layout_x, x_layout_y, min_dist):        
        # Compute new objective and power density
        x_objective = _get_objective(x_layout_x, x_layout_y, fmodel, yaw_angles, use_value)
        x_power_density = _get_power_density(fmodel, x_poly_outer)
        
        
        # If removing the turbine improves both AEP and power density, keep it
        if x_objective >= current_objective*(1 - aep_sacrifice) and x_power_density >= current_power_density:
            #factor = x_objective/current_objective
            shrink_success = True
            #print("-----------------------------------------------------------------")
            #print("Shrink happened in x-direction")
            #print(f"Shrink factor: {factor}")
            #print("-----------------------------------------------------------------")
            current_layout_x = x_layout_x
            current_layout_y = x_layout_y
            current_poly_outer = x_poly_outer
            current_objective = x_objective
            current_power_density = x_power_density
            
            
    
    # Try shrinking boundary in y direction
    y_polygons = []
    for poly in current_poly_outer.geoms:
        y_coords = [
            (
                x,
                farm_centroid.y + (y - farm_centroid.y) * shrink_factor
            )
            for x, y in poly.exterior.coords
        ]
        y_polygons.append(Polygon(y_coords))
    
    y_poly_outer = MultiPolygon(y_polygons)
    
    # Move turbines toward the centroid in x direction
    y_layout_x = current_layout_x
    y_layout_y = [farm_centroid.y + (y - farm_centroid.y) * shrink_factor for y in current_layout_y]
    
    # Check if new layout still meets minimum distance constraint
    if test_min_dist(y_layout_x, y_layout_y, min_dist):        
        # Compute new objective and power density
        y_objective = _get_objective(y_layout_x, y_layout_y, fmodel, yaw_angles, use_value)
        y_power_density = _get_power_density(fmodel, y_poly_outer)
        
        
        # If removing the turbine improves both AEP and power density, keep it
        if y_objective >= current_objective*(1 - aep_sacrifice) and y_power_density >= current_power_density:
            shrink_success = True
            #factor = y_objective/current_objective
            #print("-----------------------------------------------------------------")
            #print("Shrink happened in y-direction")
            #print(f"Shrink factor: {factor}")
            #print("-----------------------------------------------------------------")
            
            current_layout_x = y_layout_x
            current_layout_y = y_layout_y
            current_poly_outer = y_poly_outer
            current_objective = y_objective
            current_power_density = y_power_density
            
    
    new_layout_x = current_layout_x
    new_layout_y = current_layout_y
    new_poly_outer = current_poly_outer
    new_objective = current_objective
    new_power_density = current_power_density
    
    return new_layout_x, new_layout_y, new_poly_outer, new_objective, new_power_density, shrink_success


def _load_local_floris_object(
    fmodel_dict,
    wind_data=None,
):
    # Load local FLORIS object
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(wind_data=wind_data)
    return fmodel

def test_min_dist(layout_x, layout_y, min_dist):
    coords = np.array([layout_x,layout_y]).T
    dist = pdist(coords)
    return dist.min() >= min_dist

def test_point_in_bounds(test_x, test_y, poly_outer):
    return poly_outer.contains(Point(test_x, test_y))

# Return in MW
def _get_objective(
        layout_x,
        layout_y,
        fmodel,
        yaw_angles=None,
        use_value=False
):
    fmodel.set(
        layout_x=layout_x,
        layout_y=layout_y,
        yaw_angles=yaw_angles
    )
    fmodel.run()

    return fmodel.get_farm_AVP() if use_value else fmodel.get_farm_AEP()

def _get_power_density(
        fmodel,
        poly_outer,
        ):

    farm_power = fmodel.get_expected_farm_power()
    
    area = poly_outer.area
    power_density = farm_power/area
    return power_density

def _gen_dist_based_init(
    N, # Number of turbins to place
    step_size, #m, courseness of search grid
    poly_outer, # Polygon of outer boundary
    min_x,
    max_x,
    min_y,
    max_y,
    s
):
    """
    Generates an initial layout by randomly placing
    the first turbine than placing the remaining turbines
    as far as possible from the existing turbines.
    """

    # Set random seed
    np.random.seed(s)

    # Choose the initial point randomly
    init_x = float(np.random.randint(int(min_x),int(max_x)))
    init_y = float(np.random.randint(int(min_y),int(max_y)))
    while not (poly_outer.contains(Point([init_x,init_y]))):
        init_x = float(np.random.randint(int(min_x),int(max_x)))
        init_y = float(np.random.randint(int(min_y),int(max_y)))

    # Intialize the layout arrays
    layout_x = np.array([init_x])
    layout_y = np.array([init_y])
    layout = np.array([layout_x, layout_y]).T

    # Now add the remaining points
    for i in range(1,N):

        print("Placing turbine {0} of {1}.".format(i, N))
        # Add a new turbine being as far as possible from current
        max_dist = 0.
        for x in np.arange(min_x, max_x,step_size):
            for y in np.arange(min_y, max_y,step_size):
                if poly_outer.contains(Point([x,y])):
                    test_dist = cdist([[x,y]],layout)
                    min_dist = np.min(test_dist)
                    if min_dist > max_dist:
                        max_dist = min_dist
                        save_x = x
                        save_y = y

        # Add point to the layout
        layout_x = np.append(layout_x,[save_x])
        layout_y = np.append(layout_y,[save_y])
        layout = np.array([layout_x, layout_y]).T

    # Return the layout
    return layout_x, layout_y

class CodesignOptimizationRandomSearch_Sequential(LayoutOptimization):
    def __init__(
        self,
        fmodel,
        boundaries,
        min_dist=None,
        min_dist_D=None,
        distance_pmf=None,
        n_individuals=4,
        seconds_per_iteration=60.,
        total_optimization_seconds=600.,
        interface="multiprocessing",  # Options are 'multiprocessing', 'mpi4py', None
        max_workers=None,
        grid_step_size=100.,
        relegation_number=1,
        enable_geometric_yaw=False,
        use_dist_based_init=True,
        random_seed=None,
        use_value=False,
        aep_sacrifice = 0
    ):
        """
        Optimize layout using genetic random search algorithm. Details of the algorithm can be found
        in Sinner and Fleming, 2024: https://dx.doi.org/10.1088/1742-6596/2767/3/032036

        Args:
            fmodel (_type_): _description_
            boundaries (iterable(float, float)): Pairs of x- and y-coordinates
                that represent the boundary's vertices (m).
            min_dist (float, optional): The minimum distance to be maintained
                between turbines during the optimization (m). If not specified,
                initializes to 2 rotor diameters. Defaults to None.
            min_dist_D (float, optional): The minimum distance to be maintained
                between turbines during the optimization, specified as a multiple
                of the rotor diameter.
            distance_pmf (dict, optional): Probability mass function describing the
                length of steps in the random search. Specified as a dictionary with
                keys "d" (array of step distances, specified in meters) and "p"
                (array of probability of occurrence, should sum to 1). Defaults to
                uniform probability between 0.5D and 2D, with some extra mass
                to encourage large changes.
            n_individuals (int, optional): The number of individuals to use in the
                optimization. Defaults to 4.
            seconds_per_iteration (float, optional): The number of seconds to
                run each step of the optimization for. Defaults to 60.
            total_optimization_seconds (float, optional): The total number of
                seconds to run the optimization for. Defaults to 600.
            interface (str): Parallel computing interface to leverage. Recommended is 'concurrent'
                or 'multiprocessing' for local (single-system) use, and 'mpi4py' for high
                performance computing on multiple nodes. Defaults to 'multiprocessing'.
            max_workers (int): Number of parallel workers, typically equal to the number of cores
                you have on your system or HPC.  Defaults to None, which will use all
                available cores.
            grid_step_size (float): The coarseness of the grid used to generate the initial layout.
                Defaults to 100.
            relegation_number (int): The number of the lowest performing individuals to be replaced
                with new individuals generated from the best performing individual.  Must
                be less than n_individuals / 2.  Defaults to 1.
            enable_geometric_yaw (bool): Use geometric yaw code to determine approximate wake
                steering yaw angles during layout optimization routine. Defaults to False.
            use_dist_based_init (bool): Generate initial layouts automatically by placing turbines
                as far apart as possible.
            random_seed (int or None): Random seed for reproducibility. Defaults to None.
            use_value (bool, optional): If True, the layout optimization objective
                is to maximize annual value production using the value array in the
                FLORIS model's WindData object. If False, the optimization
                objective is to maximize AEP. Defaults to False.
        """
        self.yaw_angles = fmodel.core.farm.yaw_angles # Saves the yaw angles for use
        self.aep_sacrifice = aep_sacrifice
        
        # The parallel computing interface to use
        if interface == "mpi4py":
            import mpi4py.futures as mp
            self._PoolExecutor = mp.MPIPoolExecutor
        elif interface == "multiprocessing":
            import multiprocessing as mp
            self._PoolExecutor = mp.Pool
            if max_workers is None:
                max_workers = mp.cpu_count()
        elif interface is None:
            if n_individuals > 1 or (max_workers is not None and max_workers > 1):
                print(
                    "Parallelization not possible with interface=None. "
                    +"Reducing n_individuals to 1 and ignoring max_workers."
                )
                self._PoolExecutor = None
                max_workers = None
                n_individuals = 1

        # elif interface == "concurrent":
        #     from concurrent.futures import ProcessPoolExecutor
        #     self._PoolExecutor = ProcessPoolExecutor
        else:
            raise ValueError(
                f"Interface '{interface}' not recognized. "
                "Please use ' 'multiprocessing' or 'mpi4py'."
            )

        # Store the max_workers
        self.max_workers = max_workers

        # Store the interface
        self.interface = interface

        # Set and store the random seed
        self.random_seed = random_seed

        # Confirm the relegation_number is valid
        if relegation_number > n_individuals / 2:
            raise ValueError("relegation_number must be less than n_individuals / 2.")
        self.relegation_number = relegation_number

        # Store the rotor diameter and number of turbines
        self.D = fmodel.core.farm.rotor_diameters.max()
        if not all(fmodel.core.farm.rotor_diameters == self.D):
            self.logger.warning("Using largest rotor diameter for min_dist_D and distance_pmf.")
        self.N_turbines = fmodel.n_turbines

        # Make sure not both min_dist and min_dist_D are defined
        if min_dist is not None and min_dist_D is not None:
            raise ValueError("Only one of min_dist and min_dist_D can be defined.")

        # If min_dist_D is defined, convert to min_dist
        if min_dist_D is not None:
            min_dist = min_dist_D * self.D

        super().__init__(
            fmodel,
            boundaries,
            min_dist=min_dist,
            enable_geometric_yaw=enable_geometric_yaw,
            use_value=use_value,
        )
        if use_value:
            self._obj_name = "value"
            self._obj_unit = ""
        else:
            self._obj_name = "AEP"
            self._obj_unit = "[GWh]"

        # Save min_dist_D
        self.min_dist_D = self.min_dist / self.D

        # Process and save the step distribution
        self._process_dist_pmf(distance_pmf)

        # Store the Core dictionary
        self.fmodel_dict = self.fmodel.core.as_dict()

        # Save the grid step size
        self.grid_step_size = grid_step_size

        # Save number of individuals
        self.n_individuals = n_individuals

        # Store the initial locations
        self.x_initial = self.fmodel.layout_x
        self.y_initial = self.fmodel.layout_y

        # Store the total optimization seconds
        self.total_optimization_seconds = total_optimization_seconds

        # Store the seconds per iteration
        self.seconds_per_iteration = seconds_per_iteration

        # Get the initial objective value
        self.x = self.x_initial # Required by _get_geoyaw_angles
        self.y = self.y_initial # Required by _get_geoyaw_angles
        self.objective_initial = _get_objective(
            self.x_initial,
            self.y_initial,
            self.fmodel,
            self.yaw_angles,
            self.use_value,
        )
        self.power_density_initial = _get_power_density(
            self.fmodel,
            self._boundary_polygon,
            )

        # Initialize the objective statistics
        #self.objective_mean = self.objective_initial
        #self.objective_median = self.objective_initial
        #self.objective_max = self.objective_initial
        #self.objective_min = self.objective_initial
        
        # Initialize the power density statistics
        #self.power_density_mean = self.power_density_initial
        #self.power_density_median = self.power_density_initial
        #self.power_density_max = self.power_density_initial
        #self.power_density_min = self.power_density_initial

        # Initialize the numpy arrays which will hold the candidate layouts
        # these will have dimensions n_individuals x N_turbines
        #self.x_candidate = np.zeros((self.n_individuals, self.N_turbines))
        #self.y_candidate = np.zeros((self.n_individuals, self.N_turbines))
        self.x_candidate = [np.array(self.x_initial) for _ in range(self.n_individuals)]
        self.y_candidate = [np.array(self.y_initial) for _ in range(self.n_individuals)]
        self.yaw_angles_candidate = [np.array(self.yaw_angles) for _ in range(self.n_individuals)]

        self._boundary_polygon_candidate = [None] * self.n_individuals

        # Initialize the array which will hold the objective function values for each candidate
        self.objective_candidate = np.zeros(self.n_individuals)
        # Also for power density
        self.power_density_candidate = np.zeros(self.n_individuals)

        # Initialize the iteration step
        self.iteration_step = -1

        # Initialize the optimization time
        #self.opt_time_start = timerpc()
        #self.opt_time = 0

        # Generate the initial layouts
        if use_dist_based_init:
            self._generate_initial_layouts()
        else:
            #print(f'Using supplied initial layout for {self.n_individuals} individuals.')
            for i in range(self.n_individuals):
                #self.x_candidate[i,:] = self.x_initial
                #self.y_candidate[i,:] = self.y_initial
                self.objective_candidate[i] = self.objective_initial
                self.power_density_candidate[i] = self.power_density_initial
                self._boundary_polygon_candidate[i] = self._boundary_polygon    

        # Evaluate the initial optimization step
        self._evaluate_opt_step()

        # Delete stored x and y to avoid confusion
        del self.x, self.y

        # Set up to run in normal mode
        self.debug = False

    def describe(self):
        print("Random Layout Optimization")
        print(f"Number of turbines to optimize = {self.N_turbines}")
        print(f"Minimum distance between turbines = {self.min_dist_D} [D], {self.min_dist} [m]")
        print(f"Number of individuals = {self.n_individuals}")
        print(f"Seconds per iteration = {self.seconds_per_iteration}")
        print(f"Initial {self._obj_name} = {self.objective_initial/1e9:.1f} {self._obj_unit}")

    def _process_dist_pmf(self, dist_pmf):
        """
        Check validity of pmf and assign default if none provided.
        """
        if dist_pmf is None:
            jump_dist = np.min([self.xmax-self.xmin, self.ymax-self.ymin])/2
            jump_prob = 0.05

            d = np.append(np.linspace(0.0, 2.0*self.D, 99), jump_dist)
            p = np.append((1-jump_prob)/len(d)*np.ones(len(d)-1), jump_prob)
            p = p / p.sum()
            dist_pmf = {"d":d, "p":p}

        # Check correct keys are provided
        if not all(k in dist_pmf for k in ("d", "p")):
            raise KeyError("distance_pmf must contains keys \"d\" (step distance)"+\
                " and \"p\" (probability of occurrence).")

        # Check entries are in the correct form
        if not hasattr(dist_pmf["d"], "__len__") or not hasattr(dist_pmf["d"], "__len__")\
            or len(dist_pmf["d"]) != len(dist_pmf["p"]):
            raise TypeError("distance_pmf entries should be numpy arrays or lists"+\
                " of equal length.")

        if not np.isclose(dist_pmf["p"].sum(), 1):
            print("Probability mass function does not sum to 1. Normalizing.")
            dist_pmf["p"] = np.array(dist_pmf["p"]) / np.array(dist_pmf["p"]).sum()

        self.distance_pmf = dist_pmf

    def _evaluate_opt_step(self):

        # Sort the candidate layouts by objective function value and power density
        
        
        # Identify the best objective and best Power Density indices
        best_objective_index = np.argmax(self.objective_candidate)
        best_power_density_index = np.argmax(self.power_density_candidate)
        
        # Normalize AEP and Power Density using min-max scaling
        objective_min, objective_max = np.min(self.objective_candidate), np.max(self.objective_candidate)
        power_density_min, power_density_max = np.min(self.power_density_candidate), np.max(self.power_density_candidate)
        
        objective_normalized = (self.objective_candidate - objective_min) / (objective_max - objective_min + 1e-8)  # Avoid division by zero
        power_density_normalized = (self.power_density_candidate - power_density_min) / (power_density_max - power_density_min + 1e-8)
        
        # Compute combined score (equal weight simplification)
        combined_score = objective_normalized + power_density_normalized
        
        # Sort by combined score in descending order
        sorted_indices = np.argsort(combined_score)[::-1]
    
        #sorted_indices = np.argsort(self.objective_candidate)[::-1] # Decreasing order
        
        # Ensure best AEP and best Power Density are NOT in the last `relegation_number` positions
        relegation_zone = sorted_indices[-self.relegation_number:]  # Last `relegation_number` indices

        # Check if best AEP or best Power Density indices are in the relegation zone
        if best_objective_index in relegation_zone or best_power_density_index in relegation_zone:
            # Extract safe indices (not in relegation zone)
            safe_indices = list(sorted_indices[:-self.relegation_number])  # Convert to list for mutability
            relegation_zone = list(relegation_zone)  # Ensure mutability
            best_indices_to_move = set()  # Use set to avoid duplicates

            # Identify best indices in the relegation zone
            if best_objective_index in relegation_zone:
                best_indices_to_move.add(best_objective_index)
            if best_power_density_index in relegation_zone:
                best_indices_to_move.add(best_power_density_index)

            # Move the last elements from the safe zone to the relegation zone before appending best indices
            num_to_move = len(best_indices_to_move)
            to_relegation_zone = safe_indices[-num_to_move:]
            safe_indices = safe_indices[:-num_to_move]
            relegation_zone += to_relegation_zone  # Use + for list concatenation

            # Move best indices to the safe zone
            safe_indices += list(best_indices_to_move)  # Convert set to list
            relegation_zone = [idx for idx in relegation_zone if idx not in best_indices_to_move]

            # Construct new sorted list
            new_sorted_indices = safe_indices + relegation_zone
            sorted_indices = np.array(new_sorted_indices)


        
        self.objective_candidate = self.objective_candidate[sorted_indices]
        self.power_density_candidate = self.power_density_candidate[sorted_indices]
        #self.x_candidate = self.x_candidate[sorted_indices]
        #self.y_candidate = self.y_candidate[sorted_indices]
        
        # Sorting lists using list comprehension
        self.x_candidate = [self.x_candidate[i] for i in sorted_indices]
        self.y_candidate = [self.y_candidate[i] for i in sorted_indices]
        self._boundary_polygon_candidate =[self._boundary_polygon_candidate[i] for i in sorted_indices]
        self.yaw_angles_candidate = [self.yaw_angles_candidate[i] for i in sorted_indices]

        # Update the optimization time
        #self.opt_time = timerpc() - self.opt_time_start

        # Update the optimizations step
        self.iteration_step += 1

        # Update the objective statistics
        #self.objective_mean = np.mean(self.objective_candidate)
        #self.objective_median = np.median(self.objective_candidate)
        #self.objective_max = np.max(self.objective_candidate)
        #self.objective_min = np.min(self.objective_candidate)
        
        # And also for power density
        #self.power_density_mean = np.mean(self.power_density_candidate)
        #self.power_density_median = np.median(self.power_density_candidate)
        #self.power_density_max = np.max(self.power_density_candidate)
        #self.power_density_min = np.min(self.power_density_candidate)

        # Report the results
        #objective_increase_mean = (
        #    100 * (self.objective_mean - self.objective_initial) / self.objective_initial
        #)
        #objective_increase_median = (
        #    100 * (self.objective_median - self.objective_initial) / self.objective_initial
        #)
        #objective_increase_max = 100 * (self.objective_max - self.objective_initial) / self.objective_initial
        #objective_increase_min = 100 * (self.objective_min - self.objective_initial) / self.objective_initial
        #print("=======================================")
        #print(f"Optimization step {self.iteration_step:+.1f}")
        #print(f"Optimization time = {self.opt_time:+.1f} [s]")
        #print(
        #    f"Mean {self._obj_name} = {self.objective_mean/1e9:.1f}"
        #    f" {self._obj_unit} ({objective_increase_mean:+.2f}%)"
        #)
        #print(
        #    f"Median {self._obj_name} = {self.objective_median/1e9:.1f}"
        #    f" {self._obj_unit} ({objective_increase_median:+.2f}%)"
        #)
        #print(
        #    f"Max {self._obj_name} = {self.objective_max/1e9:.1f}"
        #    f" {self._obj_unit} ({objective_increase_max:+.2f}%)"
        #)
        #print(
        #    f"Min {self._obj_name} = {self.objective_min/1e9:.1f}"
        #    f" {self._obj_unit} ({objective_increase_min:+.2f}%)"
        #)
        #
        # Report the results
        #power_density_increase_mean = (
        #    100 * (self.power_density_mean - self.power_density_initial) / self.power_density_initial
        #)
        #power_density_increase_median = (
        #    100 * (self.power_density_median - self.power_density_initial) / self.power_density_initial
        #)
        #power_density_increase_max = 100 * (self.power_density_max - self.power_density_initial) / self.power_density_initial
        #power_density_increase_min = 100 * (self.power_density_min - self.power_density_initial) / self.power_density_initial
        #print(
        #    f"Mean Power Density = {self.power_density_mean:.1f}"
        #    f" [W/m^2] ({power_density_increase_mean:+.2f}%)"
        #)
        #print(
        #    f"Median Power Density = {self.power_density_median:.1f}"
        #    f" [W/m^2] ({power_density_increase_median:+.2f}%)"
        #)
        #print(
        #    f"Max Power Density = {self.power_density_max:.1f}"
        #    f" [W/m^2] ({power_density_increase_max:+.2f}%)"
        #)
        #print(
        #    f"Min Power Density = {self.power_density_min:.1f}"
        #    f" [W/m^2] ({power_density_increase_min:+.2f}%)"
        #)
        #
        #print("=======================================")

        # Replace the relegation_number worst performing layouts with relegation_number
        # best layouts
        if self.relegation_number > 0:
            #self.objective_candidate[-self.relegation_number:] = (
                #self.objective_candidate[:self.relegation_number]
            #)
            #self.x_candidate[-self.relegation_number:] = self.x_candidate[:self.relegation_number]
            #self.y_candidate[-self.relegation_number:] = self.y_candidate[:self.relegation_number]
            #self._boundary_polygon_candidate[-self.relegation_number:] = self._boundary_polygon_candidate[:self.relegation_number]
            for i in range(1, self.relegation_number + 1):
                self.objective_candidate[-i] = self.objective_candidate[i - 1]
                self.power_density_candidate[-i] = self.power_density_candidate[i - 1]
                self.x_candidate[-i] = self.x_candidate[i - 1] # Ensure deep copy
                self.y_candidate[-i] = self.y_candidate[i - 1]
                self._boundary_polygon_candidate[-i] = self._boundary_polygon_candidate[i - 1]
                self.yaw_angles_candidate[-i] = self.yaw_angles_candidate[i - 1]


    # Private methods
    def _generate_initial_layouts(self):
        """
        This method generates n_individuals initial layout of turbines. It does
        this by calling the _generate_random_layout method within a multiprocessing
        pool.
        """

        # Set random seed for initial layout
        if self.random_seed is None:
            multi_random_seeds = [None]*self.n_individuals
        else:
            multi_random_seeds = [23 + i for i in range(self.n_individuals)]
            # 23 is just an arbitrary choice to ensure different random seeds
            # to the evaluation code

        print(f'Generating {self.n_individuals} initial layouts...')
        t1 = timerpc()
        # Generate the multiargs for parallel execution
        multiargs = [
            (self.N_turbines,
            self.grid_step_size,
            self._boundary_polygon,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            multi_random_seeds[i])
            for i in range(self.n_individuals)
        ]

        if self._PoolExecutor: # Parallelized
            with self._PoolExecutor(self.max_workers) as p:
                # This code is not currently necessary, but leaving in case implement
                # concurrent later, based on parallel_computing_interface.py
                if (self.interface == "mpi4py") or (self.interface == "multiprocessing"):
                        out = p.starmap(_gen_dist_based_init, multiargs)
        else: # Parallelization not activated
            out = [_gen_dist_based_init(*multiargs[0])]

        # Unpack out into the candidate layouts
        for i in range(self.n_individuals):
            self.x_candidate[i, :] = out[i][0]
            self.y_candidate[i, :] = out[i][1]

        # Get the objective function values for each candidate layout
        for i in range(self.n_individuals):
            self.objective_candidate[i] = _get_objective(
                self.x_candidate[i, :],
                self.y_candidate[i, :],
                self.fmodel,
                self.yaw_angles,
                self.use_value,
            )

        t2 = timerpc()
        print(f"  Time to generate initial layouts: {t2-t1:.3f} s")

    def _get_initial_and_final_locs(self):
        x_initial = self.x_initial
        y_initial = self.y_initial
        x_opt = self.x_opt
        y_opt = self.y_opt
        return x_initial, y_initial, x_opt, y_opt

    def _initialize_optimization(self):
        """
        Set up logs etc
        """
        #print(f'Optimizing using {self.n_individuals} individuals.')
        print("--")
        self._opt_start_time = timerpc()
        self._opt_stop_time = self._opt_start_time + self.total_optimization_seconds

        self.objective_candidate_log = [self.objective_candidate.copy()]
        self.power_density_candidate_log = [self.power_density_candidate.copy()]
        
        self.num_objective_calls_log = []
        self._num_objective_calls = [0]*self.n_individuals

    def _run_optimization_generation(self):
        """
        Run a generation of the outer genetic algorithm
        """
        # Set random seed for the main loop
        if self.random_seed is None:
            multi_random_seeds = [None]*self.n_individuals
        else:
            multi_random_seeds = [55 + self.iteration_step + i
                for i in range(self.n_individuals)]
        # 55 is just an arbitrary choice to ensure different random seeds
        # to the initialization code

        # Update the optimization time
        #sim_time = timerpc() - self._opt_start_time
        #print(f'Optimization time: {sim_time:.1f} s / {self.total_optimization_seconds:.1f} s')


        # Generate the multiargs for parallel execution of single individual optimization
        multiargs = [
            (self.seconds_per_iteration,
                self.objective_candidate[i],
                self.power_density_candidate[i],
                self.x_candidate[i],
                self.y_candidate[i],
                self.fmodel_dict,
                self.fmodel.wind_data,
                self.min_dist,
                self._boundary_polygon_candidate[i],
                self.distance_pmf,
                self.enable_geometric_yaw,
                multi_random_seeds[i],
                self.use_value,
                self.debug,
                self.yaw_angles_candidate[i],
                self.aep_sacrifice
            )
                for i in range(self.n_individuals)
        ]

        # Run the single individual optimization in parallel
        if self._PoolExecutor: # Parallelized
            with self._PoolExecutor(self.max_workers) as p:
                out = p.starmap(_single_individual_opt, multiargs)
        else: # Parallelization not activated
            out = [_single_individual_opt(*multiargs[0])]
    
        
        

        # Unpack the results
        for i in range(self.n_individuals):
            self.objective_candidate[i] = out[i][0]
            #self.x_candidate[i] = out[i][1]
            #self.y_candidate[i] = out[i][2]
            self.x_candidate[i] = np.array(out[i][1])  # Handle dynamic size
            self.y_candidate[i] = np.array(out[i][2])
            self._num_objective_calls[i] = out[i][3]
            self._boundary_polygon_candidate[i] = out[i][4]
            self.yaw_angles_candidate[i] = out[i][5]
            self.power_density_candidate[i] = out[i][6]
        self.objective_candidate_log.append(self.objective_candidate)
        self.power_density_candidate_log.append(self.power_density_candidate)
        self.num_objective_calls_log.append(self._num_objective_calls)

        # Evaluate the individuals for this step
        self._evaluate_opt_step()

    def _finalize_optimization(self):
        """
        Package and print final results.
        """
        

        # Finalize the result
        self.objective_final = self.objective_candidate[0]
        self.power_density_final = self.power_density_candidate[0]
        self.x_opt = self.x_candidate[0]
        self.y_opt = self.y_candidate[0]
        self.yaw_angles_opt = self.yaw_angles_candidate[0]
        self.x_opt, self.y_opt, self.boundaries_opt = self._convert_to_boundaries_and_move(
            self.x_opt,
            self.y_opt,
            self._boundary_polygon_candidate[0]
            )

        # Print the final result
        objective_increase = 100 * (self.objective_final - self.objective_initial) / self.objective_initial
        print(
            f"Iteration {self._obj_name} = {self.objective_final/1e9:.1f}"
            f" {self._obj_unit} ({objective_increase:+.2f}%)"
        )
        power_density_increase = 100 * (self.power_density_final - self.power_density_initial) / self.power_density_initial
        print(
            f"Iteration Power Density = {self.power_density_final:.1f}"
            f" [W/m^2] ({power_density_increase:+.2f}%)"
        )
        
    def _convert_to_boundaries_and_move(self, layout_x, layout_y, poly_outer):
        """
        Converts a MultiPolygon object to a list of boundary coordinates and moves the boundary and turbines.
        
        Args:
            layout_x (list): X-coordinates of turbines.
            layout_y (list): Y-coordinates of turbines.
            poly_outer (MultiPolygon): The MultiPolygon object representing the boundary.

        Returns:
            tuple: (Updated layout_x, Updated layout_y, Updated boundaries)
        """
        # Move the boundary and turbines such that the first vertex of the boundary is at (0,0)
        first_vertex = list(poly_outer.geoms[0].exterior.coords)[0]
        move_x, move_y = first_vertex

        moved_boundaries = []
        for poly in poly_outer.geoms:
            moved_coords = [(x - move_x, y - move_y) for x, y in poly.exterior.coords]
            moved_boundaries.extend(moved_coords)

        moved_layout_x = [x - move_x for x in layout_x]
        moved_layout_y = [y - move_y for y in layout_y]

        return moved_layout_x, moved_layout_y, moved_boundaries

    def _test_optimize(self):
        """
        Perform a fixed number of iterations with a single worker for
        debugging and testing purposes.
        """
        # Set up a minimal problem to run on a single worker
        print("Running test optimization on a single worker.")
        self._PoolExecutor = None
        self.max_workers = None
        self.n_individuals = 1
        self.debug = True

        self._initialize_optimization()

        # Run 2 generations
        for _ in range(2):
            self._run_optimization_generation()

        self._finalize_optimization()

        return self.objective_final, self.x_opt, self.y_opt

    # Public methods
    def optimize(self):
        """
        Perform the optimization
        """
        self._initialize_optimization()

        # Run generations until the overall stop time
        while timerpc() < self._opt_stop_time:
            self._run_optimization_generation()
            #x_opt, y_opt, boundaries = self._convert_to_boundaries_and_move(self.x_candidate[0], self.y_candidate[0], self._boundary_polygon_candidate[0])
            #self.turbines_layout_positions(x_opt, y_opt, self.D, self.min_dist_D, final=True, boundaries=boundaries)

        self._finalize_optimization()

        return self.objective_final, self.power_density_final, self.x_opt, self.y_opt, self.boundaries_opt, self.yaw_angles_opt

    # Helpful visualizations
    def plot_distance_pmf(self, ax=None):
        """
        Tool to check the used distance pmf.
        """

        if ax is None:
            _, ax = plt.subplots(1,1)

        ax.stem(self.distance_pmf["d"], self.distance_pmf["p"], linefmt="k-")
        ax.grid(True)
        ax.set_xlabel("Step distance [m]")
        ax.set_ylabel("Probability")

        return ax
    
    def plot_progress_power_density(self, ax=None):

        if not hasattr(self, "power_density_candidate_log"):
            raise NotImplementedError(
                "plot_progress_power_density not yet configured for "+self.__class__.__name__
            )

        if ax is None:
            _, ax = plt.subplots(1,1)

        power_density_log_array = np.array(self.power_density_candidate_log)

        if len(power_density_log_array.shape) == 1: # Just one AEP candidate per step
            ax.plot(np.arange(len(power_density_log_array)), power_density_log_array, color="k")
        elif len(power_density_log_array.shape) == 2: # Multiple AEP candidates per step
            for i in range(power_density_log_array.shape[1]):
                ax.plot(
                    np.arange(len(power_density_log_array)),
                    power_density_log_array[:,i],
                    color="lightgray"
                )

        ax.scatter(
            np.zeros(power_density_log_array.shape[1]),
            power_density_log_array[0,:],
            color="b",
            label="Initial"
        )
        ax.scatter(
            power_density_log_array.shape[0]-1,
            power_density_log_array[-1,:].max(),
            color="r",
            label="Final"
        )

        # Plot aesthetics
        ax.grid(True)
        ax.set_xlabel("Optimization step [-]")
        ax.set_ylabel("Power Density [W/m^2]")
        ax.legend()
        
        plt.show()

        return ax
    
    def turbines_layout_positions(self, x_positions, y_positions, D, min_dist_D, final=False, boundaries=None):
        """
        This function plot the wind turbines and the boundary.
        The axis of the plot are in terms of x/d and y/d (rotor diameter)

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
        final : bool, optional
            If final=False means that this call is not for the final layout and simply plots the location and boundary
            If final=True means that this call is for the final layout so it also plot the minimum distance constaint
            as a circle around each wind turbine.
            The default is False.
        boundaries : list, optional
            The boundaries of the domain.
            If boundaries is input the plot also plots the boundary
            The default is None.

        Returns
        -------
        None.

        """
        
        fig, ax = plt.subplots(figsize=(9, 6))
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

        # Choose different markers for initial and final positions
        marker = "s" if not final else "o"
        color = "b" if not final else "r"
        ax.plot(x_positions, y_positions, marker=marker, linestyle="None", color=color)

        # Add circles around turbines for final layout
        if final:
            min_dist_radius = min_dist_D / 2  # Already in terms of D
            for x, y in zip(x_positions, y_positions):
                circle = plt.Circle((x, y), min_dist_radius, color='gray', fill=False, linestyle=":")
                ax.add_patch(circle)
      
        ax.set_xlim(min(boundary_x) - min_dist_D,max(boundary_x) + min_dist_D)
        ax.set_ylim(min(boundary_y) - min_dist_D,max(boundary_y) + min_dist_D)

        ax.set_xlabel("x (D)")
        ax.set_ylabel("y (D)")
        ax.grid(True)

        plt.show()    



def _single_individual_opt(
    seconds_per_iteration,
    initial_objective,
    initial_power_density,
    layout_x,
    layout_y,
    fmodel_dict,
    wind_data,
    min_dist,
    poly_outer,
    dist_pmf,
    enable_geometric_yaw,
    s,
    use_value,
    debug,
    yaw_angles,
    aep_sacrifice
):
    # Set random seed
    np.random.seed(s)
    
    schrink_factor_percent = 0.5
    shrink_factor=(1 - schrink_factor_percent/100)

    # Initialize the optimization time
    single_opt_start_time = timerpc()
    stop_time = single_opt_start_time + seconds_per_iteration

    num_objective_calls = 0

    # Get the fmodel
    fmodel_ = _load_local_floris_object(fmodel_dict, wind_data)

    # Initialize local variables
    num_turbines = len(layout_x)
    get_new_point = True # Will always be true, due to hardcoded use_momentum
    
    current_objective = initial_objective
    current_power_density = initial_power_density
    current_yaw_angles = yaw_angles.copy()
    current_poly_outer = poly_outer
    current_yaw_angles = yaw_angles.copy() if yaw_angles is not None else None

    # Establish geometric yaw optimizer, if desired
    #if enable_geometric_yaw:
        #yaw_opt = YawOptimizationGeometric(
            #fmodel_,
            #minimum_yaw_angle=-30.0,
            #maximum_yaw_angle=30.0,
        #)
    #else: # yaw_angles will always be none
        #yaw_angles = None

    # We have a beta feature to maintain momentum, i.e., if a move improves
    # the objective, we try to keep moving in that direction. This is currently
    # disabled.
    use_momentum = False

    # Special handling for debug mode
    if debug:
        debug_iterations = 100
        stop_time = np.inf
        dd = 0

    # Loop as long as we've not hit the stop time
    while timerpc() < stop_time:

        if debug and dd >= debug_iterations:
            break
        elif debug:
            dd += 1

        if not use_momentum:
            get_new_point = True

        if get_new_point: #If the last test wasn't successful

            # Randomly select a turbine to nudge
            tr = np.random.randint(0,num_turbines)

            # Randomly select a direction to nudge in (uniform direction)
            rand_dir = np.random.uniform(low=0.0, high=2*np.pi)

            # Randomly select a distance to travel according to pmf
            rand_dist = np.random.choice(dist_pmf["d"], p=dist_pmf["p"])

        # Get a new test point
        test_x = layout_x[tr] + np.cos(rand_dir) * rand_dist
        test_y = layout_y[tr] + np.sin(rand_dir) * rand_dist

        # In bounds?
        if not test_point_in_bounds(test_x, test_y, current_poly_outer):
            get_new_point = True
            continue

        # Make a new layout
        original_x = layout_x[tr]
        original_y = layout_y[tr]
        layout_x[tr] = test_x
        layout_y[tr] = test_y

        # Acceptable distances?
        if not test_min_dist(layout_x, layout_y,min_dist):
            # Revert and continue
            layout_x[tr] = original_x
            layout_y[tr] = original_y
            get_new_point = True
            continue

        # Does it improve the objective?
        #if enable_geometric_yaw: # Select appropriate yaw angles
            #yaw_opt.fmodel_subset.set(layout_x=layout_x, layout_y=layout_y)
            #df_opt = yaw_opt.optimize()
            #yaw_angles = np.vstack(df_opt['yaw_angles_opt'])
            
        

        num_objective_calls += 1
        test_objective = _get_objective(layout_x, layout_y, fmodel_, current_yaw_angles, use_value)
        test_power_density = _get_power_density(fmodel_, current_poly_outer)

        if test_objective >= current_objective and test_power_density >= current_power_density:
            # Accept the change
            current_objective = test_objective
            current_power_density = test_power_density
    
            # If not a random point this cycle and it did improve things
            # try not getting a new point
            # Feature is currently disabled by use_momentum flag
            get_new_point = False

        else:
            # Revert the change
            layout_x[tr] = original_x
            layout_y[tr] = original_y
            #poly_outer = original_poly_outer
            get_new_point = True
        
        # Attempt to shrink the boundary & turbines
        shrink_x, shrink_y, shrink_poly_outer, shrink_objective, shrink_power_density, shrink_success = _shrink_boundary_and_turbines(
            layout_x,
            layout_y,
            fmodel_,
            current_poly_outer,
            min_dist,
            shrink_factor,
            current_yaw_angles,
            use_value,
            current_objective,
            current_power_density,
            aep_sacrifice
            )
        if shrink_success:
            layout_x = shrink_x
            layout_y = shrink_y
            current_poly_outer = shrink_poly_outer
            current_objective = shrink_objective
            current_power_density = shrink_power_density
            get_new_point = False
        
            

    # Return the best result from this individual
    return current_objective, layout_x, layout_y, num_objective_calls, current_poly_outer, current_yaw_angles, current_power_density
