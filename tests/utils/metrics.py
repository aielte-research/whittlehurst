import numpy as np
from typing import List, Tuple

def calc_dev(
    goals: List[List[float]], 
    results: List[List[float]], 
    range_a: float, 
    range_b: float, 
    steps: int, 
    measure_interval: float
) -> Tuple[List[float], List[List[float]], List[List[float]], List[float], List[float]]:
    """
    Calculate local deviations and biases using a sliding window approach over a specified range.

    Parameters
    ----------
    goals : List[List[float]]
        Each inner list contains goal values for a particular trial.
    results : List[List[float]]
        Each inner list contains result values corresponding to the goals.
    range_a : float
        Lower bound of the x-range.
    range_b : float
        Upper bound of the x-range.
    steps : int
        Number of points in the x-range.
    measure_interval : float
        Half-window width used for selecting data points around each x value.

    Returns
    -------
    x_range : List[float]
        The x values over which the sliding window calculations were performed.
    deviations_lst : List[List[float]]
        Local deviation (standard deviation) values computed for each trial along the x-range.
    biases_lst : List[List[float]]
        Local bias (average) values computed for each trial along the x-range.
    deviation_aucs : List[float]
        Aggregated absolute deviation values (area under the curve approximation) for each trial.
    bias_aucs : List[float]
        Aggregated absolute bias values (area under the curve approximation) for each trial.
    """
    deviations_lst = []
    biases_lst = []
    deviation_aucs = []
    bias_aucs = []
    x_range = list(np.linspace(range_a, range_b, num=steps))

    for X_, Y_ in zip(goals, results):
        permut = np.argsort(X_)
        X = np.array(X_)[permut]
        Y = np.array(Y_)[permut]
        deviations = []
        biases = []
        deviation_aucs.append(0.0)
        bias_aucs.append(0.0)
        
        # Compute differences (result - goal) for each data point.
        zero_centered = [i - t for t, i in zip(X, Y)]

        for xval in x_range:
            start_idx = np.searchsorted(X, xval - measure_interval)
            end_idx = np.searchsorted(X, xval + measure_interval, side='right')
            values = zero_centered[start_idx:end_idx]

            dev = np.std(values)
            deviations.append(dev)
            deviation_aucs[-1] += abs(dev)

            avg = sum(values) / len(values) if len(values) > 0 else 0.0
            biases.append(avg)
            bias_aucs[-1] += abs(avg)
        
        # Multiply by the step size to approximate the area under the curve.
        step_size = x_range[1] - x_range[0]
        deviation_aucs[-1] *= step_size
        bias_aucs[-1] *= step_size

        deviations_lst.append(deviations)
        biases_lst.append(biases)
    return x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs

def calc_rmse(
    goals: List[List[float]], 
    results: List[List[float]], 
    range_a: float, 
    range_b: float, 
    steps: int, 
    measure_interval: float
) -> Tuple[List[float], List[List[float]], List[float]]:
    """
    Calculate local RMSE values over sliding windows and compute a global RMSE for each trial.

    Parameters
    ----------
    goals : List[List[float]]
        List of goal values for each trial.
    results : List[List[float]]
        List of result values for each trial.
    range_a : float
        Lower bound of the x-range.
    range_b : float
        Upper bound of the x-range.
    steps : int
        Number of points in the x-range.
    measure_interval : float
        Half-window width used to select data points around each x value.

    Returns
    -------
    x_range : List[float]
        The x-range used for sliding window calculations.
    rmse_lst : List[List[float]]
        Local RMSE values computed for each trial along the x-range.
    global_rmse : List[float]
        Global RMSE value computed for each trial.
    """
    rmse_lst = []
    global_rmse = []
    
    # Define the x_range based on the provided interval and number of steps.
    x_range = list(np.linspace(range_a, range_b, num=steps))
    
    # Loop over each pair of goals and results.
    for X_, Y_ in zip(goals, results):
        # Sort data based on goal values.
        permut = np.argsort(X_)
        X = np.array(X_)[permut]
        Y = np.array(Y_)[permut]
        
        # Compute differences: result minus goal.
        diffs = np.array(Y) - np.array(X)
        
        # Compute local RMSE values using sliding windows.
        local_rmse = []
        for xval in x_range:
            start = np.searchsorted(X, xval - measure_interval)
            end = np.searchsorted(X, xval + measure_interval, side='right')
            if end > start:
                window_diffs = diffs[start:end]
                rmse_val = np.sqrt(np.mean(np.square(window_diffs)))
            else:
                rmse_val = 0.0
            local_rmse.append(rmse_val)
        
        rmse_lst.append(local_rmse)
        # Compute global RMSE for the current trial.
        global_rmse.append(np.sqrt(np.mean(np.square(diffs))))
    
    return x_range, rmse_lst, global_rmse