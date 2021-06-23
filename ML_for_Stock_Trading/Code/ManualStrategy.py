"""
Proj. requirements:
- ManualStrategy.py Code implementing a ManualStrategy object (your Manual Strategy).
- It should implement testPolicy() which returns a trades data frame (see below). 
- The main part of this code should call marketsimcode as necessary to generate the plots used in the report. 
- You may use data from other symbols (such as SPY) to inform both your Manual Learner and Strategy Learner.
- The in-sample period is January 1, 2008 to December 31, 2009.
- The out-of-sample/testing period is January 1, 2010 to December 31 2011.
- ManualStrategy and StrategyLearner: Commission: $9.95, Impact: 0.005 (unless stated otherwise in an experiment).
- Plot:
    Benchmark: Green line
    Performance of Manual Strategy: Red line
    Both should be normalized to 1.0 at the start.
    Vertical blue lines indicating LONG entry points.
    Vertical black lines indicating SHORT entry points.
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data
from marketsimcode import compute_portvals
from indicators import *

def author():
    return 'kgao47'

def calculate_return(portvals):
    dr=(portvals/portvals.shift(1))-1
    dr=dr.dropna()
    
    #sharpe_ratio=np.sqrt(252)*dr.mean()/dr.std()
    avg_daily_ret=dr.mean()
    std_daily_ret=dr.std()
    cum_ret = (portvals[-1] / portvals[0]) - 1
    
    #print('--- Portfolio statistics --- ')
    print(f"Daily return mean : {avg_daily_ret}")
    print(f"Daily return std : {std_daily_ret}")
    print(f"Cumulative return : {cum_ret}")
    
def testPolicy(sym, sd, ed, sv):

    # ---> prepare indicators
    
    dates = pd.date_range(sd, ed)
    prices = get_data([sym], dates)
    df_indicators = create_indicators(prices, sym, window=14)
    A = df_indicators['price_over_sma']
    B = df_indicators['PPO_Hist']
    C = df_indicators['CCI']
    
    # ---> create trade dataframe
    
    position = 0 # 1-> Long position; -1 -> Short position; 0 -> No holding
    
    df_trades = pd.DataFrame(index=prices.index)
    df_trades = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])
    
    long_date = []
    short_date = []
    
    long_exit_date = []
    short_exit_date = []

    for i in np.arange(0, prices.shape[0]):
        date = prices.index[i]
        
        # --- determine what the combined signal is 
      
        combine = 0
        
        signal_a = 0
        if A[i] < 0.9:
            signal_a = 1 # possible buy
        elif A[i] > 1.1:
            signal_a = -1 # possible sell
        
        signal_b = 0
        if B[i] < -1:
            signal_b = 1 # possible buy
        elif B[i] > 1:
            signal_b = -1 # possible sell 
            
        signal_c = 0
        if C[i] < -100:
            signal_c = 1 # possible buy
        elif C[i] > 100:
            signal_c = -1 # possible sell
            
        combine = signal_a + signal_b + signal_c
        
        # --- make transaction 
        
        if position == 0: # currently no stock
            if combine >=2 : # enter long
                df_trades.loc[date,:] = [sym,'BUY',1000]
                position = 1
                long_date.append(prices.index[i].date())
                
            elif combine <=-2: # enter short
                df_trades.loc[date,:] = [sym,'SELL',1000]
                position = -1
                short_date.append(prices.index[i].date())
     
        elif position == 1: # currently in long position
            if combine == -2:
                df_trades.loc[date,:] = [sym,'SELL',1000]
                position = 0
                long_exit_date.append(prices.index[i].date())
                
            elif combine == -3: # strong sell signal
                df_trades.loc[date,:] = [sym,'SELL',2000]
                position = -1
                long_exit_date.append(prices.index[i].date())
                short_date.append(prices.index[i].date())
                
        elif position == -1: # currently in short position
            if combine == 2:
                df_trades.loc[date,:] = [sym,'BUY',1000]
                position = 0
                short_exit_date.append(prices.index[i].date())
            elif combine == 3: # strong buy signal
                df_trades.loc[date,:] = [sym,'BUY',2000]
                position = 1
                short_exit_date.append(prices.index[i].date())
                long_date.append(prices.index[i].date())
                 
    df_trades = df_trades.dropna()
   
    #print(df_trades)
    return df_trades, long_date, short_date, long_exit_date, short_exit_date

def test_sample(sd, ed, label, sym='JPM', commission=9.95, impact=0.005, sv=100000):
         
    # --- step 1: create trade dataframe
    ms_trades, long_date, short_date, long_exit, short_exit = testPolicy(sym, sd ,ed, sv)
    
    # --- step 2: create benchmark trade dataframe
    prices = get_data([sym], pd.date_range(sd, ed))
    bm_trades = pd.DataFrame(index=prices.index)
    bm_trades = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])
    bm_trades.loc[prices.index[0],:] = [sym, 'BUY', 1000]
    bm_trades.loc[prices.index[-1],:] = [sym, 'BUY', 0]
    
    #  --- step 3: get stat. 
    ms_port = compute_portvals(ms_trades, sd, ed, sv, commission, impact)
    bm_port = compute_portvals(bm_trades, sd, ed, sv, commission, impact)   
    
    print('--------- MS ' + label)
    calculate_return(ms_port)
    print('--------- Benchmark ' + label)
    calculate_return(bm_port)
    
    # --- step 4: make figure
    
    ft = 16
    ft1 = 12
    
    plt.close()
    
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    
    ax.plot(ms_port.index,  ms_port/ms_port[0], label="Manual Strategy", color='r', lw=2)
    ax.xaxis_date    
    ax.plot(bm_port.index,  bm_port/bm_port[0], label="Benchmark", color='g', lw=2)
    
    plt.xticks(fontsize=ft1)
    plt.yticks(fontsize=ft1)
    plt.xlabel('Date', fontsize=ft)
    plt.ylabel('Normalized Value', fontsize=ft)
    plt.legend(loc=0, fontsize=ft)
    plt.title(sym + " " + label, fontsize=ft)
    
    for date in long_date:
        ax.axvline(date,color="b",lw=1) # Vertical blue lines indicating LONG entry points
    for date in short_date:
        ax.axvline(date,color="k",lw=1) # Vertical black lines indicating SHORT entry points
    
    # !!! needs to be commented out !!!
    #for date in long_exit:
    #    ax.axvline(date,color="b",lw=1,ls='--') 
    #for date in short_exit:
    #    ax.axvline(date,color="k",lw=1,ls='--') 
    
    #plt.show()
    plt.savefig('figure_MS_'+label+'.png', dpi=200, bbox_inches='tight')
    plt.close()
    
def test_code():

    sd_in = dt.datetime(2008,1,1)
    ed_in = dt.datetime(2009,12,31)
    
    sd_out = dt.datetime(2010,1,1)
    ed_out = dt.datetime(2011,12,31)
    
    sym_sel = 'JPM'
    test_sample(sd_in, ed_in, 'In-sample', sym_sel)
    test_sample(sd_out, ed_out, 'Out-sample', sym_sel)
    
if __name__ == "__main__":
    test_code()
    
