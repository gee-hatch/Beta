# Install package
!pip install yfinance

# Import libraries
import yfinance as yf
import pandas as pd

# Import price data for Apple
data = yf.download('AAPL', start="2020-05-18", end="2020-06-18")

# Calculate adjustment factor
adjustment_factor = data['Adj Close'] / data['Close']

# Calculate adjusted open price
data['Adj Open'] = adjustment_factor * data['Open']

# Calculate adjusted high price
data['Adj High'] = adjustment_factor * data['High']

# Calculate adjusted low price
data['Adj Low'] = adjustment_factor * data['Low']

# Delete Volume column
del data['Close']
del data['Volume']
del data['Open']
del data['High']
del data['Low']
data