
def compute_trading_signal(data):
    # Create the 'trading_signal' column
    data['trading_signal'] = 0  # Default to 0
    data['holding'] = 0  # Default to 0
    
    # Initialize variables to track the last non-zero trading signal
    last_non_zero_signal = 0
    
    # Apply conditions to set +1, 0, -1 to 'trading_signal'
    for index, row in data.iterrows():
        if row['y']:
            # Check if the last non-zero signal was -1
            if last_non_zero_signal == -1:
                data.at[index, 'trading_signal'] = 1
            else:
                data.at[index, 'holding'] = 1
            last_non_zero_signal = 1
        elif not row['y']:
            # Check if the last non-zero signal was 1
            if last_non_zero_signal == 1:
                data.at[index, 'trading_signal'] = -1
            last_non_zero_signal = -1
    
    return data
