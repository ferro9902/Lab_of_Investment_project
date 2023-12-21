import pandas as pd
import numpy as np

def compute_ema(dataset, window=10):
    """
    It returns the Exponential moving average series of the day before.

    Parameters
    ----------
    dataset: Dataframe
        The Dataframe with 'date' and 'price column'
    window: int
        the time period chosen

    Returns
    -------
    Column to add to a dataset
    """
    # Ensure the dataset is sorted by date
    dataset = dataset.sort_values(by='date')

    # Calculate the smoothing factor (k)
    k = 2 / (window + 1)

    # Compute EMA using the formula for the day before
    ema_column_name = 'EMA_{}'.format(window)
    dataset[ema_column_name] = dataset['price'].shift(1).ewm(span=window, adjust=False).mean()

    return dataset[ema_column_name]

def compute_macd(dataset, ema_window_short=12, ema_window_long=26):
    """
    It returns the Moving Average Convergence Divergence series of the day before.

    Parameters
    ----------
    dataset: Dataframe
        The Dataframe with 'date' and 'price column'
    ema_window_short: int
        the time period chosen for the short EMA
    ema_window_long: int
        the time period chosen for the long EMA

    Returns
    -------
    Column to add to a dataset
    """
    # Compute the EMA for the short and long windows
    ema_short = compute_ema(dataset, window=ema_window_short)
    ema_long = compute_ema(dataset, window=ema_window_long)

    # Compute MACD as the difference between short EMA and long EMA
    macd_column_name = 'MACD'
    dataset[macd_column_name] = ema_short - ema_long

    return dataset[macd_column_name]

def compute_log_return(dataset):
    
    """
    It returns the Logarithmic returns series of the day before.

    Parameters
    ----------
    dataset: Dataframe
        The Dataframe with 'date' and 'price column'
    
    Returns
    -------
    Column to add to a dataset
    """
    
    # Ensure the dataset is sorted by date
    dataset = dataset.sort_values(by='date')

    # Compute the logarithmic return for the day before
    Log_column_name = 'Log_Return'
    dataset[Log_column_name] = np.log(dataset['price'] / dataset['price'].shift(1))

    return dataset[Log_column_name]