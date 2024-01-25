import pandas as pd
import numpy as np

def compute_ema(dataset, window=10):
    """
    It returns the Exponential moving average series.

    Parameters
    ----------
    dataset: Dataframe
        The Dataframe with 'date' and 'price column'
    window: int
        the decay chosen in terms of span

    Returns
    -------
    Column to add to a dataset
    """
    
    # Compute EMA using the formula
    EMA = dataset['price'].ewm(span=window, adjust=False).mean()

    return EMA

def compute_macd(dataset, ema_window_short=12, ema_window_long=26):
    """
    It returns the Moving Average Convergence Divergence series.

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
    MACD = ema_short - ema_long

    return MACD

def compute_log_return(dataset):
    
    """
    It returns the Logarithmic returns series.

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

    # Compute the logarithmic return
    Log_Return = np.log(dataset['price'] / dataset['price'].shift(1))

    return Log_Return