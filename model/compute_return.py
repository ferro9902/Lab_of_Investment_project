def calculate_cumulative_return(data, start_date, end_date):
    # Ensure the data is sorted by date
    data = data.sort_values(by='date')
    
    # Filter data based on start_date and end_date
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    
    # Initialize variables
    position = 0  # 0: No position, 1: Long position, -1: Short position
    initial_cash = 1.0  # Initial cash amount
    cash = initial_cash  # Current cash amount
    shares_held = 0  # Number of shares currently held
    transaction_cost = 0.001  # 1% transaction cost
    
    # Calculate cumulative returns
    for i in range(len(data)):
        signal = data['trading_signal'].iloc[i]
        price = data['price'].iloc[i]
        
        # Check if a position needs to be opened or closed
        if signal == 1 and position != 1:  # Buy signal
            # Sell current position (if any)
            cash -= shares_held * price * (1 + transaction_cost)
            
            # Buy shares
            shares_held = cash / price
            cash = 0
            position = 1
        elif signal == -1 and position != -1:  # Sell signal
            # Buy back current position (if any)
            cash += shares_held * price * (1 - transaction_cost)
            
            # Sell shares
            shares_held = 0
            position = -1
        elif signal == 0 and position != 0:  # No signal
            # Close current position (if any)
            cash += shares_held * price * (1 - transaction_cost) if position == -1 else shares_held * price * (1 + transaction_cost)
            
            # Reset position
            shares_held = 0
            position = 0
        
        # Update cash value for each time period
        cash *= (1 - transaction_cost)  # Apply transaction cost
        
    # Calculate final value
    final_value = cash + shares_held * data['price'].iloc[-1]
    
    # Calculate cumulative return
    cumulative_return = (final_value - initial_cash) / initial_cash * 100
    
    return cumulative_return

