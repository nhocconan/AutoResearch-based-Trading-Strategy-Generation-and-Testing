#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "QuantNomad - MA Strategy - 1 minute - ETHUSD"
timeframe = "1m"
leverage = 1

def generate_signals(prices):
    # Ensure prices is a DataFrame for consistent column access
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
    
    close = prices['close']
    length = 15
    
    # Calculate Simple Moving Average
    ma = close.rolling(window=length).mean()
    
    # Previous values for crossover detection
    prev_close = close.shift(1)
    prev_ma = ma.shift(1)
    
    # Crossover: Close crosses above MA (Long Signal)
    long_condition = (close > ma) & (prev_close <= prev_ma)
    
    # Crossunder: Close crosses below MA (Short Signal)
    short_condition = (close < ma) & (prev_close >= prev_ma)
    
    # Initialize signal series
    signals = pd.Series(0, index=prices.index)
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    # Maintain position state (forward fill non-zero signals)
    positions = signals.replace(0, np.nan).ffill().fillna(0)
    
    # Shift by 1 to avoid lookahead (execute on next bar open)
    positions = positions.shift(1).fillna(0)
    
    # Return numpy array matching input length
    return positions.values
