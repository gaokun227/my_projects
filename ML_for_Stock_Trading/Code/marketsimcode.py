import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'kgao47'

def compute_portvals(df_orders, sd, ed, start_val = 1000000, commission=0.00, impact=0.00):
    
    # --- step 1
    # create a look up table that has price of all stocks in the portfolio on each day

    #df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    df_orders = df_orders.sort_index()
    #df_orders.index = pd.to_datetime(df_orders.index)
    #all_dates = pd.date_range(df_orders.index.values[0], df_orders.index.values[-1])
    all_dates = dates = pd.date_range(sd, ed)
    df_stock_price = get_data(list(set(df_orders['Symbol'].values)), all_dates)

    # --- step 2
    # create a dataframe that contains number of shares on each day
    df_share = df_stock_price.copy()
    df_share.iloc[:,:] = 0.0
    
    df_share['cash'] = np.zeros(df_share.shape[0])
    df_share['cash'][0] = start_val # initialize cash
    #print(df_share['cash'][0])

    # --- step 3
    # loop over df_order to extract info on the change of stock unit and cash on the dates with transactions

    for date, order in df_orders.iterrows():
        #print(date)
        #print(order)
        
        sym = order[0] # stock sym
        action = order[1] # BUY or SELL
        unit = order[2] # number of unit in this tranaction
        price = df_stock_price.loc[date,sym] # look up stock price at this given date

        if action == "BUY":
           df_share.loc[date,sym] += unit
           df_share.loc[date,"cash"] -= unit*price
        elif action == "SELL":
           df_share.loc[date,sym] -= unit
           df_share.loc[date,"cash"] += unit*price

        # deduct 1) commission and 2) impact from cash
        df_share.loc[date, "cash"] -= commission + unit*price*impact

    #print(df_share.head(n=5))
    #print(df_stock_price.head(n=5))
    #print(df_share['cash'][0])

    # --- step 4
    # now update all dates based on info from the last step

    for i in range(1,df_share.shape[0]):
        df_share.iloc[i,:]=df_share.iloc[i,:]+df_share.iloc[i-1,:]

    # --- step 5
    # calculate port. value
    df_stock_price['cash'] = np.ones(df_share.shape[0])
    port_value = df_stock_price * df_share
    port_value["total_val"]=port_value.sum(axis=1) # port_val = cash + current value of equities

    return port_value["total_val"]
