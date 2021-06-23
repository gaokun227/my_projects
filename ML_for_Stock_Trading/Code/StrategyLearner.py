"""
Usage
- Format:
    import StrategyLearner as sl
    learner = sl.StrategyLearner(verbose = False, impact = 0.0, commission=0.0) # constructor
    learner.add_evidence(symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
    df_trades = learner.testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # testing phase
    
- For classification, you must convert your regression learner to use mode rather than mean (RTLearner, BagLearner)
    
"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data

from marketsimcode import compute_portvals 
from indicators import * 

import RTLearner as rt 
import BagLearner as bl 

class StrategyLearner(object):

    def author(self):
        return 'kgao47'

    def __init__(self, verbose=False, impact=0., commission=0.):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.critical = 0.02 # test ?
        self.ybuy = self.critical + self.impact
        self.ysell = -self.critical - self.impact 
        self.lookback = 14 # for calculating indicators
        self.lookahead = 5 # for creating training Y
        
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":5}, bags=30, boost = False, verbose = False)

    @staticmethod
    def createX(prices, syms, lookback):
        symbol = syms[0]
        df_indicators = create_indicators(prices, symbol, lookback)
        A = df_indicators['price_over_sma']
        B = df_indicators['CCI'] 
        C = df_indicators['PPO_Hist']    
        indicators = pd.concat((A,B,C),axis=1)
        indicators.fillna(0,inplace=True)
        return indicators.values

    @staticmethod
    def create_trades(prices, Y):
        trades = prices.copy()
        trades.iloc[:,:]=0
        position = 0
        
        for i in np.arange(prices.shape[0]-1):
            if position == 0:
                if Y[i] > 0:
                    trades.iloc[i,:] = 1000 # enter long from cash
                    position = 1
                elif Y[i] < 0:
                    trades.iloc[i,:] = -1000 # enter short from cash
                    position = -1
            elif position == 1:
                if Y[i] < 0:
                    trades.iloc[i,:] = -2000 # enter short from long
                    position = -1
                elif Y[i] == 0:
                    trades.iloc[i,:] = -1000 # enter cash from long
                    position = 0
            elif position == -1:
                if Y[i] > 0:
                    trades.iloc[i,:] = 2000 # enter long from short
                    position = 1
                elif Y[i]==0:
                    trades.iloc[i,:] = 1000 # enter cash from short
                    position = 0
        if position == -1:
            trades.iloc[-1,:] = 1000
        elif position == 1:
            trades.iloc[-1,:] = -1000
            
        return trades
      
    def add_evidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):

        # --- Prepare input data 
        symbol_list = [symbol]
        prices_all = get_data(symbol_list, dates = pd.date_range(sd, ed)) 
        prices = prices_all[symbol_list]
        
        # --- Prepare training X
        trainX = self.createX(prices, symbol_list, self.lookback)
        trainX = trainX[:-self.lookahead]
        
        # --- Prepare training Y
        trainY=[]
        for i in np.arange(prices.shape[0]-self.lookahead):
            ratio = (prices.iloc[i+self.lookahead,0]-prices.iloc[i,0])/prices.iloc[i,0]
            if ratio > self.ybuy:
                trainY.append(1)
            elif ratio < self.ysell:
                trainY.append(-1)
            else:
                trainY.append(0)
        trainY=np.array(trainY)

        # --- Train learner
        self.learner.addEvidence(trainX,trainY)


    def testPolicy(self, symbol = "IBM", \
        sd = dt.datetime(2009,1,1), \
        ed = dt.datetime(2010,1,1), \
        sv = 10000):

        symbol_list = [symbol]
        prices_all = get_data(symbol_list, dates = pd.date_range(sd, ed)) 
        prices = prices_all[symbol_list] 
      
        testX = self.createX(prices, symbol_list, self.lookback)
        testY = self.learner.query(testX)
        trades = self.create_trades(prices, testY)
      
        return trades

# ===========> below for testing purposes

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

def test_code():
    
    # --- Parameters
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    
    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)
    
    symbol = "JPM"
    sv = 100000
    commission = 9.95
    impact = 0.005

    # --- Train learner
    learner = StrategyLearner(verbose=False, impact=impact, commission=commission)
    learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    
    # --- In-sample and Out-sample trades
    test_in = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    test_out = learner.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)

    trades_in = change_trades(test_in, symbol)
    trades_out = change_trades(test_out, symbol)
    
    port_in = compute_portvals(trades_in, sd_in, ed_in, sv, commission, impact)
    port_out = compute_portvals(trades_out, sd_out, ed_out, sv, commission, impact)
    
    # --- Make figures
    ft = 20
    ft1 = 16
    
    plt.close()
    
    fig = plt.figure(figsize=(12, 10))
    ax1 = plt.subplot(211)
    ax1.plot(port_in.index,  port_in/port_in[0], label=symbol+' In-sample',color='k', lw=2)
    ax1.xaxis_date    
    plt.grid()
    #plt.ylim([0.5, 3])
    plt.xticks(fontsize=ft1)
    plt.yticks(fontsize=ft1)
    plt.xlabel('Date', fontsize=ft)
    plt.ylabel('Normalized Value', fontsize=ft)
    plt.legend(loc=0, fontsize=ft)
    
    ax1 = plt.subplot(212)
    ax1.plot(port_out.index,  port_out/port_out[0], label=symbol+' Out-sample',color='k', lw=2)
    ax1.xaxis_date    
    plt.grid()
    #plt.ylim([0.5, 3])
    plt.xticks(fontsize=ft1)
    plt.yticks(fontsize=ft1)
    plt.xlabel('Date', fontsize=ft)
    plt.ylabel('Normalized Value', fontsize=ft)
    plt.legend(loc=0, fontsize=ft)
    
    #plt.show()
    plt.savefig('figure_SL_test.png', dpi=200, bbox_inches='tight')
    #plt.close()
        
if __name__=="__main__":
    test_code()
    
