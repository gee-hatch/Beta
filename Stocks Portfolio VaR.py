import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Download the data
amzn = yf.download('AMZN', '2023-11-30', '2024-11-30')
tsla = yf.download('TSLA', '2023-11-30', '2024-11-30')
aapl = yf.download('AAPL', '2023-11-30', '2024-11-30')

# Extract only closing prices
amzn_close = amzn['Close']
tsla_close = tsla['Close']
aapl_close = aapl['Close']

# Put all three closing prices together
df = pd.concat([amzn_close, tsla_close, aapl_close], axis=1)
df.columns = ['AMZN', 'TSLA', 'AAPL']

# Compute the returns
df['R1'] = -df['AMZN'].pct_change()
df['R2'] = -df['TSLA'].pct_change()
df['R3'] = -df['AAPL'].pct_change()

# Construct the portfolio returns column as a weighted sum of individual asset returns and weights
w = np.array([0.4, 0.3, 0.3])
df['Rp'] = (df[['R1', 'R2', 'R3']] * w).sum(axis=1)
df = df.drop(df.index[0])

# Compute portfolio variance and standard deviation
variance_p = df['Rp'].var() * 252
sigma_p = np.sqrt(variance_p)

# Define initial portfolio value and a standard normal multiple for 95% confidence level
W = 1000000
z = norm.ppf(0.95)

# Compute and report portfolio VaR
VaR_p = sigma_p * z * W
print('Portfolio VaR corresponding to 0.95 confidence level is ${:.2f}'.format(VaR_p))