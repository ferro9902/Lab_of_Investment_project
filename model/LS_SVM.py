import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from trading_signal import compute_trading_signal
from compute_return import calculate_cumulative_return, calculate_returns
import matplotlib.pyplot as plt

def getCBX10_Df(start_date, end_date):
    df = pd.read_csv('../data/CBX10_.csv')
    df['y'] = df['price'].pct_change().shift(-1) > 0
    df[['MACD', 'EMA_10', 'Log_Return']] = df[['MACD', 'EMA_10', 'Log_Return']].shift(1)
    df = df.dropna()
    X = df[['MACD', 'EMA_10', 'Log_Return']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    compute_trading_signal(df)

    cumulative_return = calculate_cumulative_return(df, start_date, end_date)
    print(f"Cumulative Return from {start_date} to {end_date}: {cumulative_return:.2f}%")

    df = df.sort_values(by='date').reset_index(drop=True)

    df['ls_svm_cumulative_returns'] = 0
    df['ls_svm_return'] = 0
    df.at[0, 'trading_signal'] = 1

    for index, row in df.iterrows():
        if df.at[index, 'trading_signal'] == 1:
            start_date = df.at[index, 'date']
            for i in range(index, len(df)):
                if df.at[i, 'trading_signal'] == -1:
                    end_date = df.at[i, 'date']
                    break
            trade_return = calculate_returns(df, start_date, end_date)
            if index != 0:
                # Call calculate_returns method to get the return for the current trade
                df.at[index, 'ls_svm_cumulative_returns'] = df.at[index-1, 'ls_svm_cumulative_returns'] * (1 + trade_return / 100)
            else:
                df.at[index, 'ls_svm_cumulative_returns'] = 1 * (1 + trade_return / 100)
            df.at[index, 'ls_svm_return'] = trade_return
        else:
            if index == 0:
                df.at[index, 'ls_svm_cumulative_returns'] = 0
            else:
                df.at[index, 'ls_svm_cumulative_returns'] = df.at[index-1, 'ls_svm_cumulative_returns']

    # Simulate buy-and-hold strategy
    df['buy_and_hold_returns'] = df['Log_Return']
    df['buy_and_hold_cumulative_returns'] = (1 + df['buy_and_hold_returns']).cumprod()

    return df

def getSOFIX_Df(start_date, end_date):
    df = pd.read_csv('../data/SOFIX_.csv')
    df['y'] = df['price'].pct_change().shift(-1) > 0
    df[['MACD', 'EMA_10', 'Log_Return']] = df[['MACD', 'EMA_10', 'Log_Return']].shift(1)
    df = df.dropna()
    X = df[['MACD', 'EMA_10', 'Log_Return']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    compute_trading_signal(df)

    cumulative_return = calculate_cumulative_return(df, start_date, end_date)
    print(f"Cumulative Return from {start_date} to {end_date}: {cumulative_return:.2f}%")

    df = df.sort_values(by='date').reset_index(drop=True)

    df['ls_svm_cumulative_returns'] = 0
    df['ls_svm_return'] = 0
    df.at[0, 'trading_signal'] = 1

    for index, row in df.iterrows():
        if df.at[index, 'trading_signal'] == 1:
            start_date = df.at[index, 'date']
            for i in range(index, len(df)):
                if df.at[i, 'trading_signal'] == -1:
                    end_date = df.at[i, 'date']
                    break
            trade_return = calculate_returns(df, start_date, end_date)
            if index != 0:
                # Call calculate_returns method to get the return for the current trade
                df.at[index, 'ls_svm_cumulative_returns'] = df.at[index-1, 'ls_svm_cumulative_returns'] * (1 + trade_return / 100)
            else:
                df.at[index, 'ls_svm_cumulative_returns'] = 1 * (1 + trade_return / 100)
            df.at[index, 'ls_svm_return'] = trade_return
        else:
            if index == 0:
                df.at[index, 'ls_svm_cumulative_returns'] = 0
            else:
                df.at[index, 'ls_svm_cumulative_returns'] = df.at[index-1, 'ls_svm_cumulative_returns']

    # Simulate buy-and-hold strategy
    df['buy_and_hold_returns'] = df['Log_Return']
    df['buy_and_hold_cumulative_returns'] = (1 + df['buy_and_hold_returns']).cumprod()
    
    return df

def getSP500_Df(start_date, end_date):
    df = pd.read_csv('../data/SP500_.csv')
    df['y'] = df['price'].pct_change().shift(-1) > 0
    df[['MACD', 'EMA_10', 'Log_Return']] = df[['MACD', 'EMA_10', 'Log_Return']].shift(1)
    df = df.dropna()
    X = df[['MACD', 'EMA_10', 'Log_Return']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    compute_trading_signal(df)

    cumulative_return = calculate_cumulative_return(df, start_date, end_date)
    print(f"Cumulative Return from {start_date} to {end_date}: {cumulative_return:.2f}%")

    df = df.sort_values(by='date').reset_index(drop=True)

    df['ls_svm_cumulative_returns'] = 0
    df['ls_svm_return'] = 0
    df.at[0, 'trading_signal'] = 1

    for index, row in df.iterrows():
        if df.at[index, 'trading_signal'] == 1:
            start_date = df.at[index, 'date']
            for i in range(index, len(df)):
                if df.at[i, 'trading_signal'] == -1:
                    end_date = df.at[i, 'date']
                    break
            trade_return = calculate_returns(df, start_date, end_date)
            if index != 0:
                # Call calculate_returns method to get the return for the current trade
                df.at[index, 'ls_svm_cumulative_returns'] = df.at[index-1, 'ls_svm_cumulative_returns'] * (1 + trade_return / 100)
            else:
                df.at[index, 'ls_svm_cumulative_returns'] = 1 * (1 + trade_return / 100)
            df.at[index, 'ls_svm_return'] = trade_return
        else:
            if index == 0:
                df.at[index, 'ls_svm_cumulative_returns'] = 0
            else:
                df.at[index, 'ls_svm_cumulative_returns'] = df.at[index-1, 'ls_svm_cumulative_returns']

    # Simulate buy-and-hold strategy
    df['buy_and_hold_returns'] = df['Log_Return']
    df['buy_and_hold_cumulative_returns'] = (1 + df['buy_and_hold_returns']).cumprod()
    
    return df

def getSP600_Df(start_date, end_date):
    df = pd.read_csv('../data/SP600_.csv')
    df['y'] = df['price'].pct_change().shift(-1) > 0
    df[['MACD', 'EMA_10', 'Log_Return']] = df[['MACD', 'EMA_10', 'Log_Return']].shift(1)
    df = df.dropna()
    X = df[['MACD', 'EMA_10', 'Log_Return']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    compute_trading_signal(df)

    cumulative_return = calculate_cumulative_return(df, start_date, end_date)
    print(f"Cumulative Return from {start_date} to {end_date}: {cumulative_return:.2f}%")

    df = df.sort_values(by='date').reset_index(drop=True)

    df['ls_svm_cumulative_returns'] = 0
    df['ls_svm_return'] = 0
    df.at[0, 'trading_signal'] = 1

    for index, row in df.iterrows():
        if df.at[index, 'trading_signal'] == 1:
            start_date = df.at[index, 'date']
            for i in range(index, len(df)):
                if df.at[i, 'trading_signal'] == -1:
                    end_date = df.at[i, 'date']
                    break
            trade_return = calculate_returns(df, start_date, end_date)
            if index != 0:
                # Call calculate_returns method to get the return for the current trade
                df.at[index, 'ls_svm_cumulative_returns'] = df.at[index-1, 'ls_svm_cumulative_returns'] * (1 + trade_return / 100)
            else:
                df.at[index, 'ls_svm_cumulative_returns'] = 1 * (1 + trade_return / 100)
            df.at[index, 'ls_svm_return'] = trade_return
        else:
            if index == 0:
                df.at[index, 'ls_svm_cumulative_returns'] = 0
            else:
                df.at[index, 'ls_svm_cumulative_returns'] = df.at[index-1, 'ls_svm_cumulative_returns']

    # Simulate buy-and-hold strategy
    df['buy_and_hold_returns'] = df['Log_Return']
    df['buy_and_hold_cumulative_returns'] = (1 + df['buy_and_hold_returns']).cumprod()
    
    return df

def plotDF(df):
    max_effectiveness_index = df['ls_svm_cumulative_returns'].idxmax()
    max_effectiveness_value = df.loc[max_effectiveness_index]['ls_svm_cumulative_returns']+0.2

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df['ls_svm_cumulative_returns'], label='LS-SVM Strategy')
    plt.plot(df['buy_and_hold_cumulative_returns'], label='Buy and Hold')
    plt.vlines(x=max_effectiveness_index, ymin=0.5, ymax=max_effectiveness_value, colors='red', ls=':', lw=2, label='maximum effectiveness at date ' + df.at[max_effectiveness_index, 'date'])
    plt.legend()
    plt.title('Comparison of LS-SVM Strategy and Buy and Hold BELEX15')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.show()