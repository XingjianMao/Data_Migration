import time


def measure_runtime(func, *args, **kwargs):
   
    start_time = time.time()  # Start the timer

    result = func(*args, **kwargs)  # Call the function with provided arguments

    end_time = time.time()  # End the timer

    elapsed_time = end_time - start_time  # Calculate elapsed time

    return result, elapsed_time