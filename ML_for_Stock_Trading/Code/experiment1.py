"""
Experiment 1

Proj requirement:

- For your report, trade only the symbol JPM. We will test your Strategy Learner with other symbols as well.
- You may use data from other symbols (such as SPY) to inform both your Manual Learner and Strategy Learner.
- The in-sample period is January 1, 2008 to December 31, 2009.
- The out-of-sample/testing period is January 1, 2010 to December 31 2011.
- Starting cash is $100,000.
- Allowable positions are: 1000 shares long, 1000 shares short, 0 shares.
- Compare your Manual Strategy with your Strategy Learner in-sample trading JPM. Create a chart that shows:

  Value of the ManualStrategy portfolio (normalized to 1.0 at the start)
  Value of the StrategyLearner portfolio (normalized to 1.0 at the start)
  Value of the Benchmark portfolio (normalized to 1.0 at the start)
  The code that implements this experiment and generates the relevant charts and 
  data should be submitted as experiment1.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from util import get_data, plot_data
from marketsimcode import compute_portvals

import StrategyLearner as st
import ManualStrategy as ms

def author():
    return 'kgao47'

def change_trades(trades,symbol):
    
    trades_new = pd.DataFrame(index=trades.index)
    trades_new = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])
     
    for i in np.arange(trades.shape[0]): 
        index = trades.index[i]
        if trades.loc[index,symbol] == 2000:
            trades_new.loc[index,:] = [symbol,'BUY',2000]     
        elif trades.loc[index,symbol] == 1000:
            trades_new.loc[index,:] = [symbol,'BUY',1000] 
        elif trades.loc[index,symbol] == -2000:
            trades_new.loc[index,:] = [symbol,'SELL',2000]
        elif trades.loc[index,symbol] == -1000:
            trades_new.loc[index,:] = [symbol,'SELL',1000]
                
    return trades_new

def calculate_return(portvals):
    dr=(portvals/portvals.shift(1))-1
    dr=dr.dropna()

    avg_daily_ret=dr.mean()
    std_daily_ret=dr.std()
    cum_ret = (portvals[-1] / portvals[0]) - 1
    sharp_ratio = np.sqrt(252.0)*avg_daily_ret/std_daily_ret
    
    print(f"Daily return mean : {avg_daily_ret}")
    print(f"Daily return std : {std_daily_ret}")
    print(f"Cumulative return : {cum_ret}")
    print(f"Sharp ratio : {sharp_ratio}")
    
def conduct_exp():
    
    # --- step 1: initialize parameters
    date_str = dt.datetime(2008, 1, 1)
    date_end = dt.datetime(2009, 12, 31)
    symbol = "JPM"
    sv = 100000
    commission = 9.95
    impact = 0.005

    # --- Step 2: create trade dataframe from Manual Strategy 
    ms_trades, _, _, _, _ = ms.testPolicy(symbol, date_str, date_end, sv)
    
    # --- Step 3: create trade dataframe from Strategy Learner 
    learner = st.StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner.add_evidence(symbol=symbol, sd=date_str, ed=date_end, sv=sv)
    test = learner.testPolicy(symbol=symbol, sd=date_str, ed=date_end, sv=sv)
    st_trades = change_trades(test, symbol)
    
    # --- Step 4: Create Benchmark trade dataframe 
    prices = get_data([symbol], pd.date_range(date_str, date_end))
    bm_trades = pd.DataFrame(index=prices.index)
    bm_trades = pd.DataFrame(columns=['Symbol', 'Order', 'Shares'])
    bm_trades.loc[prices.index[0],:] = [symbol, 'BUY', 1000]
    bm_trades.loc[prices.index[-1],:] = [symbol, 'BUY', 0]

    # --- Step 5: Portfolio value
    ms_port = compute_portvals(ms_trades, date_str, date_end, sv, commission, impact)
    st_port = compute_portvals(st_trades, date_str, date_end, sv, commission, impact)
    bm_port = compute_portvals(bm_trades, date_str, date_end, sv, commission, impact)
    
    # --- Step 6: Portfolio statistics
    print('-----------')
    print('Manual port. statistics:')
    calculate_return(ms_port)
    
    print('-----------')
    print('Learner port. statistics:')
    calculate_return(st_port)
    
    print('-----------')
    print('Benchmark port. statistics:')
    calculate_return(bm_port)
    
    # --- Step 7: Make plots
    ft = 20
    ft1 = 16
    
    plt.close()
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(111)
    #ax1.plot(prices.index,  prices, label='JPM',color='k', lw=1)
    ax1.plot(bm_port.index,  bm_port/bm_port[0], label='Benchmark',color='g', lw=2)
    ax1.xaxis_date
    ax1.plot(ms_port.index,  ms_port/ms_port[0], label='Manual Strategy',color='r', lw=2)
    ax1.plot(st_port.index,  st_port/st_port[0], label='Strategy Learner',color='m', lw=2)
    
    plt.grid()

    plt.ylim([0.5, 2.5])
    plt.xticks(fontsize=ft1)
    plt.yticks(fontsize=ft1)
    plt.xlabel('Date', fontsize=ft)
    plt.ylabel('Normalized Value', fontsize=ft)
    plt.legend(loc=0, fontsize=ft)
    #plt.show()
    plt.savefig('figure_MS_vs_ST.png', dpi=200, bbox_inches='tight')
    plt.close()
        
if __name__=="__main__":
    conduct_exp()

