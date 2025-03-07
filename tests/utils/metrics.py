import numpy as np

def calc_dev(goals, results, range_a, range_b, steps, measure_interval):
    """
    Calculate deviations and biases using a sliding window approach along a specified range.
    
    Parameters:
        goals (list of lists): Each inner list contains goal values for a particular trial.
        results (list of lists): Each inner list contains result values corresponding to the goals.
        range_a (float): Lower bound of the x_range.
        range_b (float): Upper bound of the x_range.
        steps (int): Number of points in the x_range.
        measure_interval (float): Half-window width used for selecting data points around each x value.
        
    Returns:
        x_range (list): The x values over which the sliding window calculations were performed.
        deviations_lst (list of lists): Local deviation (standard deviation) values computed for each trial along the x_range.
        biases_lst (list of lists): Local bias (average) values computed for each trial along the x_range.
        deviation_aucs (list): Aggregated absolute deviation values (area under the curve approximation) for each trial.
        bias_aucs (list): Aggregated absolute bias values (area under the curve approximation) for each trial.
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
        deviations=[]
        biases=[]
        deviation_aucs.append(0)
        bias_aucs.append(0)
        
        zero_centered=[i-t for t,i in zip(X,Y)]

        for xval in  x_range:
            values=zero_centered[np.searchsorted(X, xval-measure_interval):np.searchsorted(X, xval+measure_interval,side='right')]

            dev=np.std(values)
            deviations.append(dev)
            deviation_aucs[-1]+=abs(dev)

            if len(values)==0:
                avg = 0
            else:
                avg = sum(values)/len(values)
            biases.append(avg)
            bias_aucs[-1]+=abs(avg)
        deviation_aucs[-1]*=x_range[1]-x_range[0]
        bias_aucs[-1]*=x_range[1]-x_range[0]

        deviations_lst.append(deviations)
        biases_lst.append(biases)
    return x_range, deviations_lst, biases_lst, deviation_aucs, bias_aucs

def calc_rmse(goals, results, range_a, range_b, steps, measure_interval):
    """
    Calculates the local RMSE for sliding windows defined over an x_range 
    (using the goal values) and returns both the list of RMSE values for each 
    (goal, result) pair as well as the global RMSE across all differences.
    
    Parameters:
        goals (list of lists): List of goal values for each trial.
        results (list of lists): List of result values for each trial.
        range_a (float): Lower bound of the x_range.
        range_b (float): Upper bound of the x_range.
        steps (int): Number of steps (points) in the x_range.
        measure_interval (float): Half-window width used to select data points 
                                  around each xval.
    
    Returns:
        x_range (list): The x_range used for sliding window calculations.
        rmse_lst (list of lists): Local RMSE values computed for each trial along the x_range.
        global_rmse (float): The global RMSE computed over all differences in all trials.
    """
    rmse_lst = []        # List to store the sliding-window RMSEs for each (goal, result) pair.
    global_sq_errors = []  # List to accumulate squared errors for global RMSE.
    global_rmse = []
    
    # Define the x_range based on the provided interval and number of steps.
    x_range = list(np.linspace(range_a, range_b, num=steps))
    
    # Loop over each pair of goals and results.
    for X_, Y_ in zip(goals, results):
        # Sort by the goal values (X_).
        permut = np.argsort(X_)
        X = np.array(X_)[permut]
        Y = np.array(Y_)[permut]
        
        # Compute the differences (result minus goal)
        diffs = np.array(Y) - np.array(X)
        
        # Compute local RMSE for the sliding windows.
        local_rmse = []
        for xval in x_range:
            # Select indices of X within the window [xval-measure_interval, xval+measure_interval].
            start = np.searchsorted(X, xval - measure_interval)
            end = np.searchsorted(X, xval + measure_interval, side='right')
            if end > start:
                window_diffs = diffs[start:end]
                # Calculate RMSE for the window.
                rmse_val = np.sqrt(np.mean(np.square(window_diffs)))
            else:
                rmse_val = 0.0  # Set to 0 if no values are present.
            local_rmse.append(rmse_val)
        
        rmse_lst.append(local_rmse)
        # Add squared errors from the entire trial to the global list.
        global_rmse.append(np.sqrt(np.mean(np.square(diffs))))
    
    return x_range, rmse_lst, global_rmse