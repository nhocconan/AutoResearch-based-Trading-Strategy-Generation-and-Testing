#!/usr/bin/env python3
import pandas as pd
import numpy as np

name = "RSI BTC vs SPX (Adapted)"
timeframe = "1h"
leverage = 1

def calculate_rsi(close_prices, period=14):
    """Calculate RSI using Wilder's smoothing method (Pine Script compatible)."""
    if len(close_prices) < period + 1:
        return np.full(len(close_prices), np.nan)
    
    deltas = np.diff(close_prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Initialize arrays
    avg_gains = np.zeros(len(close_prices))
    avg_losses = np.zeros(len(close_prices))
    rsi = np.full(len(close_prices), np.nan)
    
    # First SMA
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    # Wilder's smoothing (RMA)
    for i in range(period + 1, len(close_prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RSI
    mask = avg_losses == 0
    rs = np.zeros_like(avg_gains)
    rs[~mask] = avg_gains[~mask] / avg_losses[~mask]
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def generate_signals(prices):
    """
    Generate trading signals based on RSI crossover logic.
    
    Args:
        prices (pd.DataFrame): Must contain 'open', 'high', 'low', 'close', 'volume'.
        
    Returns:
        np.ndarray: Array of signals (1=Long, -1=Short, 0=Neutral) with len(prices).
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Prices must be a pandas DataFrame")
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in prices.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    n = len(prices)
    signals = np.zeros(n, dtype=int)
    
    # Extract data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate BTC RSI
    btc_rsi = calculate_rsi(close, period=14)
    
    # SPX RSI Simulation
    # NOTE: Original strategy requires SPX data via request.security.
    # Repo constraints forbid external API calls and multi-asset columns.
    # We substitute SPX RSI with a constant neutral value (50) to allow execution.
    # This invalidates the correlation logic but maintains code structure.
    spx_rsi = np.full(n, 50.0)
    
    # Strategy Parameters
    tp_percent = 3.0 / 100.0
    
    # State tracking
    position = 0  # 0=Neutral, 1=Long, -1=Short
    entry_price = 0.0
    
    # Iterate through bars to generate signals for NEXT bar
    # signals[i] represents position held DURING bar i
    # Logic evaluated at close of i-1 determines signals[i]
    
    for i in range(1, n):
        # Check Take Profit exits first based on current bar's price action
        # If TP hit during bar i-1, we close position for bar i
        if position != 0:
            exited = False
            if position == 1:  # Long
                # Check if high of previous bar hit TP
                if i > 0 and high[i-1] >= entry_price * (1 + tp_percent):
                    exited = True
            elif position == -1:  # Short
                # Check if low of previous bar hit TP
                if i > 0 and low[i-1] <= entry_price * (1 - tp_percent):
                    exited = True
            
            if exited:
                position = 0
                entry_price = 0.0
        
        # Set signal for current bar i based on position state
        signals[i] = position
        
        # Check Entry Conditions based on previous bar's close (i-1)
        # Avoid lookahead: use data available at close of i-1
        if i >= 15:  # Ensure RSI is valid
            curr_btc = btc_rsi[i-1]
            prev_btc = btc_rsi[i-2]
            curr_spx = spx_rsi[i-1]
            prev_spx = spx_rsi[i-2]
            
            if not np.isnan(curr_btc) and not np.isnan(prev_btc):
                # Long Condition: BTC RSI crosses over SPX RSI
                if prev_btc <= prev_spx and curr_btc > curr_spx:
                    if position == 0:
                        position = 1
                        entry_price = close[i-1]  # Enter at next open approx
                
                # Short Condition: BTC RSI crosses under SPX RSI
                elif prev_btc >= prev_spx and curr_btc < curr_spx:
                    if position == 0:
                        position = -1
                        entry_price = close[i-1]
    
    return signals

if __name__ == "__main__":
    # Example usage stub
    print(f"Strategy: {name}")
    print(f"Timeframe: {timeframe}")
