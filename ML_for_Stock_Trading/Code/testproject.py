import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from util import get_data, plot_data
from marketsimcode import compute_portvals

import StrategyLearner as st
import ManualStrategy as ms

import experiment1 as exp1
import experiment1 as exp2

def author():
    return 'kgao47'

if __name__ == "__main__":
    ms.test_code() 
    st.test_code()
    exp1.conduct_exp()
    exp2.conduct_exp()
    
    
