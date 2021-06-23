"""
Proj requirements

Conduct an experiment with your StrategyLearner that shows how changing the value of impact 
should affect in-sample trading behavior (use at least two metrics). 

Trade JPM on the in-sample period with a commission of $0.00.
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
    
    return avg_daily_ret, cum_ret, sharp_ratio
    
def conduct_exp():
    
    # --- step 1: initialize parameters
    date_str = dt.datetime(2008, 1, 1)
    date_end = dt.datetime(2009, 12, 31)
    symbol = "JPM"
    sv = 100000
    commission = 0 #9.95
    impact_list = [0, 0.005, 0.015, 0.025, 0.05]
    color_list = ['b', 'g', 'orange', 'r', 'm']
        
    port_list = []
    adr_list = []
    cr_list = []
    sr_list = []

    # --- Step 2: create trade dataframe from Strategy Learner 
        
    # Note impact is considered in both learner and compute_portvals !!!
    
    for impact in impact_list:
        
        learner = st.StrategyLearner(verbose=False, impact=impact, commission=commission)
        learner.add_evidence(symbol=symbol, sd=date_str, ed=date_end, sv=sv)
        test = learner.testPolicy(symbol=symbol, sd=date_str, ed=date_end, sv=sv)
        st_trades = change_trades(test, symbol)
        st_port = compute_portvals(st_trades, date_str, date_end, sv, commission, impact)
        adr, cr, sr = calculate_return(st_port)
        
        port_list.append(st_port)
        adr_list.append(adr)
        cr_list.append(cr)
        sr_list.append(sr)
        
    # --- Step 3: Make plots : port value
    ft = 18
    ft1 = 12
    
    plt.close()
    
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(111)
    
    for i in np.arange(len(impact_list)):
        port = port_list[i]
        ax1.plot(port.index,  port/port[0], label='impact = '+str(impact_list[i]),color=color_list[i], lw=2)
        ax1.xaxis_date
    
    plt.grid()

    #plt.ylim([0.5, 2.5])
    plt.xticks(fontsize=ft1)
    plt.yticks(fontsize=ft1)
    plt.xlabel('Date', fontsize=ft)
    plt.ylabel('Normalized Value', fontsize=ft)
    plt.legend(loc=0, fontsize=ft)
    #plt.show()
    plt.savefig('figure_impact_port.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # --- Step 5: Make plots : two metric 
    
    fig = plt.figure(figsize=(20, 6))
    
    ax1 = plt.subplot(121)
    ax1.plot(impact_list,  cr_list, marker='o', ms = 10, color='k', lw=2, ls='-')
    plt.grid()
    ax1.axhline(y=0,color="k",lw=2, ls='--')
    plt.xticks(impact_list,fontsize=ft1)
    plt.yticks(fontsize=ft1)
    plt.xlabel('Impact factor', fontsize=ft)
    plt.ylabel('Cumulative return', fontsize=ft)
    plt.title('a) Cumulative return', fontsize=ft)

    ax1 = plt.subplot(122)
    ax1.plot(impact_list,  adr_list, marker='o', ms = 10, color='k', lw=2, ls='-')
    plt.grid()
    ax1.axhline(y=0,color="k",lw=2, ls='--')
    plt.xticks(impact_list,fontsize=ft1)
    plt.yticks(fontsize=ft1)
    plt.xlabel('Impact factor', fontsize=ft)
    plt.ylabel('Averaged daily return', fontsize=ft)
    plt.title('b) Averaged daily return', fontsize=ft)
    
    #plt.show()
    plt.savefig('figure_impact_metrics.png', dpi=200, bbox_inches='tight')
    plt.close()
       
if __name__=="__main__":
    conduct_exp()


