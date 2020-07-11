# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:49:07 2020

@author: Amogh

                                   #####  PROBLEM/TASK  #####                                 
We are creating the Market Model (Single Factor Model) for HDFC Bank using daily price data for
approximately last 5 years. We are using the NIFTY50 as proxy for the Market Portfolio.


"""


import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import datetime as dt

# Downloading price data for HDFC Bank and Nifty 50 for the last 5 years:
start = dt.datetime.today()-dt.timedelta(1800)
end = dt.datetime.today()
df = yf.download(["HDFCBANK.NS", "^NSEI"], start, end)
df = df["Close"]
df = df.rename(columns = {"^NSEI" : "NIFTY50", "HDFCBANK.NS" : "HDFCBANK"})

# Calculating and storing the daily returns in a dataframe:
ret = df.pct_change().dropna()

# Visualizing daily returns of HDFC Bank and NIFTY50:
ret.plot(x = "NIFTY50", y = "HDFCBANK", figsize = (8, 6), kind = "scatter")
plt.show()
# Comment: We can see a Positive Linear Relationship between the daily returns of HDFC Bank and Nifty50. 

# Creating the regression model:
model = ols("HDFCBANK ~ NIFTY50", data=ret)
results = model.fit()
print(results.summary())

                                     #####  COMMENTS  ######
# R-squared of 58.9% is pretty good and high for the Market Model.
# We can conclude that the Intercept is very close to 0.
# Slope coefficient of 0.95 which is also the Beta factor for HDFC Bank.
# Looking at the p-value, we can also conclude that Nifty50 returns explain the returns of HDFC Bank stock with Statistical Significance.
                                     
                                     