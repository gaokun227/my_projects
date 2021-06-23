import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data

def author():
    return 'kgao47'

def create_indicators(df_prices, sym, window=20):
    """
    Input: dataframe containing stock prices and stock name
    Output: dataframe containing various indicators
    """

    prices = df_prices[sym]
    prices = prices / prices[0] # !!! change back !!!

    df_indicators = pd.DataFrame(index=prices.index)

    # --- create indicator 1: Price/SMA
    #window = 20 # lookback period
    df_indicators['price'] = prices
    df_indicators['sma_20d'] = prices.rolling(window=window, center=False).mean()
    df_indicators['smsd_20d'] = prices.rolling(window=window, center=False).std() 
    df_indicators['price_over_sma'] = df_indicators['price']/df_indicators['sma_20d']

    # --- create indicator 2: Bollinger Band value
    df_indicators['bb_upper'] = df_indicators['sma_20d'] + (2*df_indicators['smsd_20d'])
    df_indicators['bb_lower'] = df_indicators['sma_20d'] - (2*df_indicators['smsd_20d'])
    df_indicators['percent_b'] = (df_indicators['price'] - df_indicators['bb_lower']) / (df_indicators['bb_upper']-df_indicators['bb_lower'])
    df_indicators['bb_value'] = (df_indicators['price'] - df_indicators['sma_20d'])/(2*df_indicators['smsd_20d'])
    
    # --- create indicator 3: MACD (Moving Average Convergence Divergence)
    df_indicators['ema_12d'] = df_indicators['price'].ewm(span=12, adjust=False).mean()
    df_indicators['ema_26d'] = df_indicators['price'].ewm(span=26, adjust=False).mean()
    df_indicators['MACD'] = df_indicators['ema_12d'] - df_indicators['ema_26d']
    df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
    df_indicators['MACD_Hist'] = df_indicators['MACD'] - df_indicators['MACD_Signal']
   
    # --- create indicator 4: PPO (Percentage Price Oscillator)
    df_indicators['PPO'] = 100*(df_indicators['ema_12d'] - df_indicators['ema_26d'])/df_indicators['ema_26d']
    df_indicators['PPO_Signal'] = df_indicators['PPO'].ewm(span=9, adjust=False).mean()
    df_indicators['PPO_Hist'] = df_indicators['PPO'] - df_indicators['PPO_Signal']
    
    # --- create indicator 5: CCI (Commodity Channel Index)
    
    #df_indicators['sma_20d'] = prices.rolling(window=20, center=False).mean()
    df_indicators['price_minus_sma_20d'] = np.abs(df_indicators['price'] - df_indicators['sma_20d'])
    #df_indicators['mean_deviation_20d'] = df_indicators['price_minus_sma_20d'].rolling(window=window, center=False).mean()
    mad = lambda x: np.fabs(x - x.mean()).mean()
    df_indicators['mean_deviation_20d'] = prices.rolling(window=window).apply(mad, raw=True)
    df_indicators['CCI'] = (df_indicators['price']-df_indicators['sma_20d'])/(0.015 * df_indicators['mean_deviation_20d'])
    
    return df_indicators #.dropna()

def test_code():
    
    """
    - For your report, use only the symbol JPM. This will enable us to more easily compare results.
    - Use the time period January 1, 2008 to December 31 2009.
    """

    # --- step 1 generate data
    date_str = dt.datetime(2008, 1, 1) #dt.datetime(2008, 1, 1)
    date_end = dt.datetime(2009, 12, 31) #dt.datetime(2009, 12, 31)
    all_dates = pd.date_range(date_str, date_end)

    sym = 'JPM'
    df_prices = get_data([sym], all_dates)

    # --- step 2 generate all 5 indicators
    df_indicators = create_indicators(df_prices, sym, window=10)
   
    print(df_indicators['CCI'].head(n=20))

if __name__ == "__main__":
    test_code()

