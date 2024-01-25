
def compute_trading_signal(data):
    # Create the 'trading_signal' column
    data['trading_signal'] = 0  # Default to 0
    
   # Apply conditions to set +1, 0, -1 to 'trading_signal'
    data.loc[(data['y'].shift(1) != data['y']) & (data['y'] == True), 'trading_signal'] = 1
    data.loc[(data['y'].shift(1) != data['y']) & (data['y'] == False), 'trading_signal'] = -1
    
    return data
