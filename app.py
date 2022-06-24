#%% Import Libraries

import pandas as pd
import numpy as np
import datetime as dt
from xbbg import blp
import streamlit as st
import statsmodels.api as sm
import plotly.express as px


#%% Declare Tickers 

tickers = ['AAPL US EQUITY', 
           'BABA US EQUITY', 
           'DIS US EQUITY', 
           'LULU US EQUITY',   
           'PSX US EQUITY', 
           'SPGI US EQUITY', 
           'LMT US EQUITY', 
           'TXN US EQUITY',
           'V US EQUITY',
           'MCD US EQUITY',
           'SPX INDEX']

date = '2014-12-31'

#%% BBG Data Pull

date_range = pd.bdate_range(date, dt.date.today())
returns_data_ = pd.DataFrame(index = date_range)

@st.cache
def grabData():
    dfs = []
    for t in tickers:
        df = blp.bdh(t, ['PX_LAST'], start_date = date, end_date = dt.date.today())
        df.columns = df.columns.droplevel(0)
        df.rename(columns = {'PX_LAST' : t}, inplace = True)
        df.index = pd.to_datetime(df.index)
        dfs.append(df)

    returns_data_ = pd.concat(dfs, axis=1)
    return returns_data_

returns_data_ = grabData()

#%%Copy results


returns_data = returns_data_.copy().sort_index()

returns_data.index = pd.to_datetime(returns_data.index)

rules = {'AAPL US EQUITY': 'last', 
           'BABA US EQUITY': 'last', 
           'DIS US EQUITY' : 'last', 
           'LULU US EQUITY': 'last',   
           'PSX US EQUITY' : 'last', 
           'SPGI US EQUITY' : 'last', 
           'LMT US EQUITY' : 'last', 
           'TXN US EQUITY' : 'last',
           'V US EQUITY' : 'last',
           'MCD US EQUITY' : 'last',
           'SPX INDEX' : 'last'}
         
returns_data = returns_data.resample('M').agg(rules) #Filter for monthly data
returns_data.columns = returns_data.columns.str.replace('US EQUITY', '')
share_prices = returns_data.loc[:, ~returns_data.columns.str.contains('INDEX')]

#%% Calculate Returns
def calcReturns():
    cols = returns_data.columns
    
    for col in cols:
        returns_data[col + ' RETURN'] = returns_data[col].pct_change()
        
    returns_data.dropna(inplace = True)
    return returns_data

returns_data = calcReturns()

#%% OLS Analysis

def calcBetas():
    cols_ex = list(returns_data)[11:-1]
    # groups = returns_data.groupby([returns_data.index.year])                                                                  
    years = list(returns_data.index.year.unique())
    
    betas = {}
    
    for year in years[1:]: #two year rolling betas      
        df = returns_data.loc[str(year -1):str(year)]                             
        betas_year = {}
        for col in cols_ex:
            beta = sm.OLS(df[col], df['SPX INDEX RETURN'])
            results = beta.fit() #runs the model and finds the line of best fit        
            betas_year[col] = results.params[0]
    
        betas[year] = betas_year
    
    return betas
        
betas = calcBetas()

allBetasDF = pd.DataFrame(betas).T
allBetasDF.columns = allBetasDF.columns.str.replace('RETURN', '') + 'BETA'


#%% Optimize Function

from scipy.optimize import minimize

def optw(w, V):
    # Function returns the variance of the portfolio given weights and covar matrix
    return(np.matmul(np.matmul(w,V),w))

#%% Portoflio Optimization

calc_returns = returns_data.loc[:,(returns_data.columns.str.contains('RETURN')) & (~returns_data.columns.str.contains('INDEX'))]
years = list(returns_data.index.year.unique())

variance = []
mean = []
stdev = []
ls_optimal_weights = {} 
long_optimal_weights = {}

for year in years[1:]:
    df = calc_returns.loc[str(year - 1)]  
    V = np.cov(df, rowvar = False) * 10000
    mu = np.mean(df, axis = 0) * 100 
    std = np.std(df, axis = 0) 
    n = mu.shape[0]
    
    variance.append(V)
    mean.append(mu)
    stdev.append(std)

    # Set initial values for the weights and define the expected return
    w = np.matrix([1/n] * n).T #initial guess equal weights
    expect_return = 5
    
    #lower/upper limits for the weights
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(n))
    
    constraint_1 = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    constraint_2 = {'type': 'ineq', 'fun': lambda x: np.sum(x * mu ) - expect_return}
        #eq --> means the fun has to be exactly equal to 0
        #ineq --> means the fun has to be positive
        #for constraint 2 using ineq to allow for returns to be > expect_return if possible
    
    #Negative weights allowed (short selling)
    ls_weight = minimize(optw, w, (V), constraints=(constraint_1,constraint_2), options={'maxiter':1000})

    #No negative weights (no short selling) by adding boundaries
    long_weight = minimize(optw, w, (V), constraints=(constraint_1,constraint_2),bounds=bounds,options={'maxiter':1000})

    #weight
    # ls_weight.fun #calculated minimum variance at given expected return
    # ls_weight.x #calculated optimal weights for each ticker
    
    # long_weight.fun 
    # long_weight.x 

    ls_optimal_weights[year] = ls_weight.x  
    long_optimal_weights[year] = long_weight.x

ls_optimal_weights = pd.DataFrame(ls_optimal_weights).T
long_optimal_weights = pd.DataFrame(long_optimal_weights).T

lst = ['AAPL','BABA','DIS', 'LULU', 'PSX', 'SPGI', 'LMT', 'TXN','V','MCD']

ls_optimal_weights.columns = lst 
long_optimal_weights.columns = lst

#%% Calculate Returns

#Create individual dfs for each portfolio
actual_returns = returns_data.loc[:, (~returns_data.columns.str.contains('PREDICTED')) & 
                                  (returns_data.columns.str.contains('RETURN')) & 
                                  (~returns_data.columns.str.contains('INDEX'))]
long_only_returns = actual_returns.copy()
ls_returns = actual_returns.copy()
equal_weight = actual_returns.copy()
equal_weight = equal_weight.loc[equal_weight.index.year.isin(years[1:])] #Filter to match long and l/s index 


#Cumulative single stock returns for dashboard chart
stocks = returns_data.columns[11:22]
stocks = stocks.str.replace('RETURN', '')

for stock in stocks:
    share_prices[stock + ' CUM RETURN'] = (1 + returns_data[stock + 'RETURN']).cumprod() - 1

share_prices = share_prices.loc[:, share_prices.columns.str.contains('CUM RETURN')]
share_prices.columns = share_prices.columns.str.replace('  CUM RETURN', '')


#SPX Returns DF
spx_returns = pd.DataFrame(returns_data.iloc[:,-1])
spx_returns = spx_returns.loc[spx_returns.index.year.isin(years[1:])] #Filter to match long and l/s index 
spx_returns['SPX CUM RETURN'] = (1 + spx_returns['SPX INDEX RETURN']).cumprod() - 1


#Applying weights to portfolios
long_only_returns = long_only_returns.merge(long_optimal_weights, left_on = long_only_returns.index.year, right_index = True)
ls_returns = ls_returns.merge(ls_optimal_weights, left_on = ls_returns.index.year, right_index = True)
long_only_returns.drop(columns = 'key_0', inplace = True)
ls_returns.drop(columns = 'key_0', inplace = True)

for ele in lst:
    equal_weight[ele] = n / 100 


#Calculating cumulative returns, Sharpe Ratios, Annualized Volatility

#Is there a more efficient way to do W.A across >2 columns?

#Portfolio Returns
long_only_returns['LONG PORTFOLIO RETURN'] = (long_only_returns['AAPL  RETURN'] * long_only_returns['AAPL']) + (long_only_returns['BABA  RETURN'] * long_only_returns['BABA']) + (long_only_returns['DIS  RETURN'] * long_only_returns['DIS']) + (long_only_returns['LULU  RETURN'] * long_only_returns['LULU']) + (long_only_returns['PSX  RETURN'] * long_only_returns['PSX']) + (long_only_returns['SPGI  RETURN'] * long_only_returns['SPGI']) + (long_only_returns['LMT  RETURN'] * long_only_returns['LMT']) + (long_only_returns['TXN  RETURN'] * long_only_returns['TXN']) + (long_only_returns['V  RETURN'] * long_only_returns['V']) + (long_only_returns['MCD  RETURN'] * long_only_returns['MCD'])     
ls_returns['LS PORTFOLIO RETURN'] = (ls_returns['AAPL  RETURN'] * ls_returns['AAPL']) + (ls_returns['BABA  RETURN'] * ls_returns['BABA']) + (ls_returns['DIS  RETURN'] * ls_returns['DIS']) + (ls_returns['LULU  RETURN'] * ls_returns['LULU']) + (ls_returns['PSX  RETURN'] * ls_returns['PSX']) + (ls_returns['SPGI  RETURN'] * ls_returns['SPGI']) + (ls_returns['LMT  RETURN'] * ls_returns['LMT']) + (ls_returns['TXN  RETURN'] * ls_returns['TXN']) + (ls_returns['V  RETURN'] * ls_returns['V']) + (ls_returns['MCD  RETURN'] * ls_returns['MCD'])              
equal_weight['EQW PORTFOLIO RETURN'] = (equal_weight['AAPL  RETURN'] * equal_weight['AAPL']) + (equal_weight['BABA  RETURN'] * equal_weight['BABA']) + (equal_weight['DIS  RETURN'] * equal_weight['DIS']) + (equal_weight['LULU  RETURN'] * equal_weight['LULU']) + (equal_weight['PSX  RETURN'] * equal_weight['PSX']) + (equal_weight['SPGI  RETURN'] * equal_weight['SPGI']) + (equal_weight['LMT  RETURN'] * equal_weight['LMT']) + (equal_weight['TXN  RETURN'] * equal_weight['TXN']) + (equal_weight['V  RETURN'] * equal_weight['V']) + (equal_weight['MCD  RETURN'] * equal_weight['MCD'])              

#Cumulative Returns
long_only_returns['LONG CUM RETURN'] = (1 + long_only_returns['LONG PORTFOLIO RETURN']).cumprod() - 1
ls_returns['LS CUM RETURN'] = (1 + ls_returns['LS PORTFOLIO RETURN']).cumprod() - 1
equal_weight['EQW CUM RETURN'] = (1 + equal_weight['EQW PORTFOLIO RETURN']).cumprod() - 1

#3M Rolling STD
long_only_returns['LONG 3M PORT SD'] = long_only_returns['LONG PORTFOLIO RETURN'].rolling(window = 3, min_periods = 3).std()
ls_returns['LS 3M PORT SD'] = ls_returns['LS PORTFOLIO RETURN'].rolling(window = 3, min_periods = 3).std()
equal_weight['EQW 3M PORT SD'] = equal_weight['EQW PORTFOLIO RETURN'].rolling(window = 3, min_periods = 3).std()

#3M Rolling Avg Return
long_only_returns['LONG 3M MEAN RETURN'] = long_only_returns['LONG PORTFOLIO RETURN'].rolling(window = 3, min_periods = 3).mean()
ls_returns['LS 3M MEAN RETURN'] = ls_returns['LS PORTFOLIO RETURN'].rolling(window = 3, min_periods = 3).mean()
equal_weight['EQW 3M MEAN RETURN'] = equal_weight['EQW PORTFOLIO RETURN'].rolling(window = 3, min_periods = 3).mean()

#3M Sharpe Ratio
long_only_returns['LONG 3M SHARPE RATIO'] = long_only_returns['LONG 3M MEAN RETURN'] / long_only_returns['LONG 3M PORT SD']
ls_returns['LS 3M SHARPE RATIO'] = ls_returns['LS 3M MEAN RETURN'] / ls_returns['LS 3M PORT SD']
equal_weight['EQW 3M SHARPE RATIO'] = equal_weight['EQW 3M MEAN RETURN'] / equal_weight['EQW 3M PORT SD']

#Annualized Volatility
long_only_returns['LONG VOLATILITY'] = long_only_returns['LONG 3M PORT SD'] * np.sqrt(4)
ls_returns['LS VOLATILITY'] = ls_returns['LS 3M PORT SD'] * np.sqrt(4)
equal_weight['EQW VOLATILITY'] = equal_weight['EQW 3M PORT SD'] * np.sqrt(4)

longDF = long_only_returns[['LONG PORTFOLIO RETURN', 'LONG CUM RETURN', 'LONG 3M PORT SD','LONG 3M MEAN RETURN', 'LONG 3M SHARPE RATIO', 'LONG VOLATILITY']]
lsDF = ls_returns[['LS PORTFOLIO RETURN', 'LS CUM RETURN', 'LS 3M PORT SD','LS 3M MEAN RETURN', 'LS 3M SHARPE RATIO', 'LS VOLATILITY']]
eqwDF = equal_weight[['EQW PORTFOLIO RETURN', 'EQW CUM RETURN','EQW 3M PORT SD','EQW 3M MEAN RETURN', 'EQW 3M SHARPE RATIO', 'EQW VOLATILITY']]


#Merge all ports into single DF for analysis
finalDF = longDF.merge(lsDF, left_index = True, right_index = True).merge(eqwDF, left_index = True, right_index = True).merge(spx_returns, left_index = True, right_index = True)


#%% Streamlit Dash

# import plotly.io as pio
# pio.renderers.default='browser' #to launch graphs in browser when running in Spyder
share_prices.columns[0:10]
ticker_select = st.multiselect('Select a ticker.', share_prices.columns[0:10])

start = st.date_input('Pick Your Start Date', dt.date(2016, 1, 31))
end = st.date_input('Pick Your End Date')

fig_one = px.line(share_prices.loc[start:end][ticker_select], title = 'Share Price Performance Over Time')
st.plotly_chart(fig_one)

fig_two = px.line(finalDF.loc[start:end][['LONG CUM RETURN', 'LS CUM RETURN', 'EQW CUM RETURN', 'SPX CUM RETURN']], title = 'Portfolio Cumulative Returns')
st.plotly_chart(fig_two)

fig_thr = px.line(finalDF.loc[start:end][['LONG 3M SHARPE RATIO', 'LS 3M SHARPE RATIO', 'EQW 3M SHARPE RATIO']], title = '3 Month Rolling Sharpe Ratio')
st.plotly_chart(fig_thr)

portfolio_select = st.selectbox('Select a Portfolio', ['LONG', 'LS', 'EQW'], index = 0)

fig_four = px.line(finalDF.loc[start:end][portfolio_select + ' VOLATILITY'], title = 'Annualized Volatility')
st.plotly_chart(fig_four)

st.write(allBetasDF)

portfolio_select_two = st.selectbox('Select a Portfolio', {'LONG':long_optimal_weights, 'LS': ls_optimal_weights}, index = 0)
st.write(portfolio_select_two)
