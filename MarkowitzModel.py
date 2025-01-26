import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# On Average there are 252 Trading days in a year
Trading_Days = 252

# Generate Random Portfolio with different Weights
Num_Portfolio = 10000

# Name of the Stocks in the Portfolio
stocks = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB',]
#Historical Data - Date Range

start_date = '2017-01-01'
end_date = '2025-01-01'

def download_data():
    # Key - Stock Name ; Value - Stock Value
    stock_data = {}

    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10, 5))
    plt.show()

# Calculate log Return

def calculate_return(data):
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

def show_statistics(returns):
    #Mean of Annual Return
    print(returns.mean() * Trading_Days)
    print(returns.cov() * Trading_Days)

def show_mean_variance(returns, weights):
    # Calculate Annual Return
    portfolio_return = np.sum(returns.mean() * weights) * Trading_Days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * Trading_Days, weights)))

    print("Expected Portfolio Return (Mean): ", portfolio_return)
    print("Expected Portfolio Volatility (Standard Deviation): ", portfolio_volatility)

def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

def generate_portfolios(returns):

    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(Num_Portfolio):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * Trading_Days)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * Trading_Days, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * Trading_Days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * Trading_Days, weights)))

    return np.array([portfolio_return, portfolio_volatility, portfolio_return / portfolio_volatility])

# scipy optimize module can find the minimum of a given function
# the maximum of a f(x) is the minimum of -f(x)
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

#Constraints
# The sum of weight = 1

def optimize_portfolio(weights, returns):
    # Sum of weight is 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    # the weights can be at most 1
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio: ", optimum['x'].round(3))
    print("Expected return, Volatility and Sharpe ratio: ",statistics(optimum['x'].round(3),returns))

def show_optimal_portfolios(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets/portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
    plt.show()

if __name__ == '__main__':

    dataset = download_data()
    show_data(dataset)
    log_daily_returns = calculate_return(dataset)
   # show_statistics(log_daily_returns)
    pweights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)
    optimum = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimum,log_daily_returns)
    show_optimal_portfolios(optimum, log_daily_returns, means, risks)