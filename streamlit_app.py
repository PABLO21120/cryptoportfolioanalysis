

import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

st.title('Python For Finance - Multi-Portfolio Optimization Project')
st.markdown("""
In this app, we create and compare three distinct portfolios:
1. **Equities Only** 
2. **Equities and Bonds**
3. **Equities, Bonds, and Cryptocurrencies**
Each portfolio will be optimized, and we will analyze their correlation, covariance, and efficient frontier.
""")

# Sidebar - Portfolio Configuration
st.sidebar.header("Portfolio Configuration")
available_stocks = ["AAPL", "MSFT", "AMZN", "TSLA", "GOOGL", "FB", "V", "JNJ", "WMT", "JPM"]
available_bonds = ["IEF", "LQD", "TLT", "SHY", "BNDX"]
available_crypto = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]

# Portfolio Selection
selected_stocks = st.sidebar.multiselect("Select Equities:", available_stocks, ["AAPL", "MSFT", "GOOGL"])
selected_bonds = st.sidebar.multiselect("Select Bonds:", available_bonds, ["IEF", "LQD"])
selected_crypto = st.sidebar.multiselect("Select Cryptocurrencies:", available_crypto, ["BTC-USD", "ETH-USD"])

# Date Range Selection
start_date = st.sidebar.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2023, 1, 1))

# Data Loading and Cleaning
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data.interpolate(method='linear')

# Portfolio Datasets
stocks_data = load_data(selected_stocks, start_date, end_date)
bonds_data = load_data(selected_bonds, start_date, end_date)
crypto_data = load_data(selected_crypto, start_date, end_date)

# Combine Datasets for Each Portfolio
portfolio_stocks = stocks_data
portfolio_stocks_bonds = pd.concat([stocks_data, bonds_data], axis=1)
portfolio_all = pd.concat([stocks_data, bonds_data, crypto_data], axis=1)

# Display Data
st.markdown("### Portfolio Prices")
st.write("Equities Only Portfolio:")
st.dataframe(portfolio_stocks)

st.write("Equities and Bonds Portfolio:")
st.dataframe(portfolio_stocks_bonds)

st.write("Equities, Bonds, and Cryptos Portfolio:")
st.dataframe(portfolio_all)

# Calculate Returns
returns_stocks = portfolio_stocks.pct_change().dropna()
returns_stocks_bonds = portfolio_stocks_bonds.pct_change().dropna()
returns_all = portfolio_all.pct_change().dropna()

# Portfolio Analysis
st.markdown("### Correlation and Covariance Matrices")

def plot_heatmap(data, title):
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
    plt.title(title)
    st.pyplot(fig)

plot_heatmap(returns_stocks, "Correlation Matrix - Equities Only")
plot_heatmap(returns_stocks_bonds, "Correlation Matrix - Equities and Bonds")
plot_heatmap(returns_all, "Correlation Matrix - Equities, Bonds, and Cryptos")

# Covariance Matrices
cov_stocks = returns_stocks.cov() * 252
cov_stocks_bonds = returns_stocks_bonds.cov() * 252
cov_all = returns_all.cov() * 252

st.markdown("### Covariance Matrices")
st.write("Equities Only Covariance Matrix")
st.dataframe(cov_stocks)

st.write("Equities and Bonds Covariance Matrix")
st.dataframe(cov_stocks_bonds)

st.write("Equities, Bonds, and Cryptos Covariance Matrix")
st.dataframe(cov_all)

# Portfolio Optimization
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(weights * mean_returns) * 252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std_dev

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_returns, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std_dev

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Optimize Each Portfolio
mean_returns_stocks = returns_stocks.mean()
optimal_weights_stocks = optimize_portfolio(mean_returns_stocks, cov_stocks)
mean_returns_stocks_bonds = returns_stocks_bonds.mean()
optimal_weights_stocks_bonds = optimize_portfolio(mean_returns_stocks_bonds, cov_stocks_bonds)
mean_returns_all = returns_all.mean()
optimal_weights_all = optimize_portfolio(mean_returns_all, cov_all)

st.markdown("### Optimal Portfolio Weights")
st.write("Equities Only Portfolio Weights:")
st.write(dict(zip(portfolio_stocks.columns, optimal_weights_stocks)))

st.write("Equities and Bonds Portfolio Weights:")
st.write(dict(zip(portfolio_stocks_bonds.columns, optimal_weights_stocks_bonds)))

st.write("Equities, Bonds, and Cryptos Portfolio Weights:")
st.write(dict(zip(portfolio_all.columns, optimal_weights_all)))

# Plot Efficient Frontier
st.markdown("### Efficient Frontier")

def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.01):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

    fig, ax = plt.subplots()
    ax.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o')
    ax.scatter(sdp, rp, marker='*', color='r', s=200, label='Maximum Sharpe Ratio')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend()
    st.pyplot(fig)

st.write("Efficient Frontier - Equities Only")
plot_efficient_frontier(mean_returns_stocks, cov_stocks)

st.write("Efficient Frontier - Equities and Bonds")
plot_efficient_frontier(mean_returns_stocks_bonds, cov_stocks_bonds)

st.write("Efficient Frontier - Equities, Bonds, and Cryptos")
plot_efficient_frontier(mean_returns_all, cov_all)
