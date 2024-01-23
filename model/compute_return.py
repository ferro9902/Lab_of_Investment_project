def calculate_cumulative_return(data, start_date, end_date):
    data = data.sort_values(by='date').reset_index(drop=True)
    # Ensure the data is sorted by date
    cumulative_return = 0
    current_return = 0
    data.at[0, 'trading_signal'] = 1

    for index, row in data.iterrows():
        if data.at[index, 'trading_signal'] == 1:
            start_date = data.at[index, 'date']
            for i in range(index, len(data)):
                if data.at[i, 'trading_signal'] == -1:
                    end_date = data.at[i, 'date']
                    break
            trade_return = calculate_returns(data, start_date, end_date)
            if index != 0:
                # Call calculate_returns method to get the return for the current trade
                cumulative_return = cumulative_return * (1 + trade_return / 100)
            else:
                cumulative_return = 1 * (1 + trade_return / 100)
    
    # Calculate cumulative return
    cumulative_return = (cumulative_return-1) * 100
    
    return cumulative_return

def calculate_returns(data, start_date, end_date, initial_cash=10000, commission_rate=0.005):
    """
    Calculate returns of an index bought on start_date and sold on end_date.

    Parameters:
    - data: DataFrame with columns ['date', 'price', 'MACD', 'EMA_10', 'Log_Return']
    - start_date: Buy date
    - end_date: Sell date
    - initial_cash: Initial cash available for buying the index (default: $100,000)
    - commission_rate: Commission rate for buying and selling (default: 0.5%)

    Returns:
    - Returns the calculated returns as a percentage.
    """
    # Filter data between start_date and end_date
    subset = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()

    # Calculate the number of shares bought with initial cash
    initial_shares = initial_cash / subset.iloc[0]['price']

    # Calculate the value of the investment at the end_date
    final_value = initial_shares * subset.iloc[-1]['price']

    # Calculate total commissions
    total_commissions = (initial_cash + final_value) * commission_rate

    # Calculate net return
    net_return = final_value - total_commissions - initial_cash

    # Calculate return percentage
    return_percentage = (net_return / initial_cash) * 100

    return return_percentage