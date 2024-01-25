import pandas as pd

def gross_returns(dataset, start_date, end_date):
    # Convert 'date' column to datetime format
    dataset['date'] = pd.to_datetime(dataset['date'])
    
    # Filter dataset based on start and end dates
    subsample = dataset[(dataset['date'] >= start_date) & (dataset['date'] <= end_date)].copy()
    
    # Compute daily capital gain
    subsample['daily_gain'] = (-1*subsample['price'].diff(periods=-1)) * subsample['trading_signal']
    
    # Compute cumulative capital gain %
    cumulative_gain = 100*(subsample['daily_gain'].sum() / subsample.iloc[0]['price'])
    
    # Return only the cumulative gain &
    return cumulative_gain


def net_returns(dataset, start_date, end_date):
    # Convert 'date' column to datetime format
    dataset['date'] = pd.to_datetime(dataset['date'])
    
    # Filter dataset based on start and end dates
    subsample = dataset[(dataset['date'] >= start_date) & (dataset['date'] <= end_date)].copy()
    
    # Compute daily capital gain
    subsample['daily_gain'] = (-1*subsample['price'].diff(periods=-1)) * subsample['trading_signal']
    
    # Apply 1% transaction cost
    subsample['daily_gain'] = subsample['daily_gain'] - (0.01 * abs(subsample['daily_gain']))
    
    # Compute cumulative capital gain %
    cumulative_gain = 100*(subsample['daily_gain'].sum() / subsample.iloc[0]['price'])
    
    # Return only the cumulative gain %
    return cumulative_gain

def buy_and_hold_returns(dataset, start_date, end_date):
    
    # Ensure the 'date' column is in datetime format
    dataset['date'] = pd.to_datetime(dataset['date'])
    
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter the dataset for the specified date range
    selected_data = dataset[(dataset['date'] >= start_date) & (dataset['date'] <= end_date)].copy()
    
    # Extract the initial and final prices
    initial_price = selected_data.iloc[0]['price']
    final_price = selected_data.iloc[-1]['price']
    
    # Calculate the buy and hold return
    buy_and_hold_return = 100*((final_price - initial_price) / initial_price)
    
    return buy_and_hold_return