#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1-day RSI filter + volume confirmation
# Uses 4h price breaking above/below 20-period Donchian channel, filtered by 1-day RSI (bullish >50, bearish <50)
# and volume > 1.5x median volume. Works in bull markets (long breakouts) and bear markets (short breakdowns).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[0] = 50  # First value
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 20-period Donchian channel on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + RSI > 50 (bullish bias) + volume confirmation
        if (close[i] > highest_high[i] and 
            rsi_1d_aligned[i] > 50 and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + RSI < 50 (bearish bias) + volume confirmation
        elif (close[i] < lowest_low[i] and 
              rsi_1d_aligned[i] < 50 and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout (opposite Donchian break)
        elif position == 1 and close[i] < lowest_low[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > highest_high[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_RSI_Volume"
timeframe = "4h"
leverage = 1.0