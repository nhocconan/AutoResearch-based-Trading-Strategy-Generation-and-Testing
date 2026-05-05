#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss
# Long when price breaks above 20-period high AND 1d close > 1d EMA50 (uptrend)
# Short when price breaks below 20-period low AND 1d close < 1d EMA50 (downtrend)
# Exit when price touches opposite Donchian band OR 1d trend reverses
# Uses discrete sizing (0.30) to limit fee drag. Target: 20-50 trades/year per symbol.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 1d for HTF trend to avoid counter-trend trades and 4h for entry timing.

name = "4h_Donchian20_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get daily close array for EMA
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    # Highest high of last 20 bars (including current)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 bars (including current)
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-period high AND 1d uptrend
            if (close[i] > highest_20[i] and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below 20-period low AND 1d downtrend
            elif (close[i] < lowest_20[i] and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price touches 20-period low OR 1d trend changes to downtrend
            if (close[i] < lowest_20[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price touches 20-period high OR 1d trend changes to uptrend
            if (close[i] > highest_20[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals