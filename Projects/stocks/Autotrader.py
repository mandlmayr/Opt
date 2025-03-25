# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 16:46:46 2025

@author: Michael
"""

from opti import Opt
import numpy as np
import pandas as pd
from datetime import datetime


df = pd.read_csv('sp500_stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])
a=df[df['Date']>datetime(2023,1,1)]
b=a[a['Symbol']=='BLK']