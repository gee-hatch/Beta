# Complete Bollinger Bands Trading Strategy - Fixed Version
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Step 1: Fetch data for AAPL
print("Fetching AAPL data...")
AAPL_data = yf.download('AAPL', start='2022-09-30', end='2023-10-04', auto_adjust=True)

# Fix: Handle MultiIndex columns by flattening or selecting the ticker
if isinstance(AAPL_data.columns, pd.MultiIndex):
    # If MultiIndex, flatten the columns
    AAPL_data.columns = AAPL_data.columns.droplevel(1)

df = AAPL_data.copy()

print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print("Columns:", df.columns.tolist())

# Step 2: Calculate Bollinger Bands
# Define parameters
window = 20  # The window for the moving average
std_multiplier = 2  # The number of standard deviations to use

# Calculate the rolling mean and standard deviation
df['SMA'] = df['Close'].rolling(window=window).mean()
df['Upper'] = df['SMA'] + (std_multiplier * df['Close'].rolling(window=window).std())
df['Lower'] = df['SMA'] - (std_multiplier * df['Close'].rolling(window=window).std())

# Step 3: Create Trading Signals
# Create a column to store trading signals
df['Signal'] = 0  # Initialize with no signal

# Buy signal: Price touches or crosses below the lower band
df.loc[df['Close'] <= df['Lower'], 'Signal'] = 1

# Sell signal: Price touches or crosses above the upper band
df.loc[df['Close'] >= df['Upper'], 'Signal'] = -1

# Step 4: Backtesting and Performance Metrics
# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Calculate strategy returns by multiplying signal with daily returns
df['Strategy Returns'] = df['Signal'].shift(1) * df['Returns']

# Calculate cumulative returns
df['Cumulative Returns'] = (1 + df['Strategy Returns']).cumprod()

# Check if cumulative returns are positive
if df['Cumulative Returns'].iloc[-1] > 1:
    print("Cumulative Returns are positive!")

# Calculate key performance metrics
cumulative_returns = df['Cumulative Returns'].iloc[-1] - 1
returns_mean = df['Strategy Returns'].mean()
returns_std = df['Strategy Returns'].std()
sharpe_ratio = (returns_mean / returns_std) if returns_std != 0 else 0

# Calculate maximum drawdown
rolling_max = (1 + df['Strategy Returns']).cumprod().rolling(window=window, min_periods=1).max()
daily_drawdown = (1 + df['Strategy Returns']).cumprod() / rolling_max - 1
max_drawdown = daily_drawdown.min()

# Print the results
print(f"\n=== Performance Metrics ===")
print(f"Cumulative Returns: {cumulative_returns * 100:.2f}%")
print(f"Returns Mean: {returns_mean * 100:.2f}%")
print(f"Returns Standard Deviation: {returns_std * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

# Step 5: Visualization
# Plot the price data and Bollinger Bands
plt.figure(figsize=(15, 10))

# First subplot: Price and Bollinger Bands
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Close'], label='AAPL Price', linewidth=1)
plt.plot(df.index, df['SMA'], label='SMA (20)', linestyle='--', alpha=0.7)
plt.plot(df.index, df['Upper'], label='Upper Band', linestyle='--', alpha=0.7)
plt.plot(df.index, df['Lower'], label='Lower Band', linestyle='--', alpha=0.7)

# Mark buy and sell signals
buy_signals = df[df['Signal'] == 1]
sell_signals = df[df['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=50, label='Buy Signal')
plt.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=50, label='Sell Signal')

plt.title('AAPL Bollinger Bands Trading Strategy')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Second subplot: Cumulative Returns
plt.subplot(2, 1, 2)
plt.plot(df.index, df['Cumulative Returns'], label='Strategy Cumulative Returns', color='green', linewidth=2)
plt.axhline(y=1, color='black', linestyle='--', label='Initial Investment', alpha=0.7)
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display final summary
print(f"\n=== Strategy Summary ===")
print(f"Total trades: {(df['Signal'] != 0).sum()}")
print(f"Buy signals: {(df['Signal'] == 1).sum()}")
print(f"Sell signals: {(df['Signal'] == -1).sum()}")
print(f"Final portfolio value: ${df['Cumulative Returns'].iloc[-1]:.2f} (starting from $1.00)")

# Show sample of the data
print(f"\n=== Sample Data ===")
print(df[['Close', 'SMA', 'Upper', 'Lower', 'Signal']].head(25))