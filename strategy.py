#!/usr/bin/env python3
"""
1d_1w_100EMA_Breakout
Strategy: Long-only breakout on 1d timeframe with 1-week EMA100 trend filter.
- Long when price breaks above 1-week high and price is above 1-week EMA100 (bullish trend)
- Exit when price breaks below 1-week low (stop loss)
- Position size: 0.30
- Uses 1d timeframe as primary and 1w for trend confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter and breakout levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA100 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema100_1w = close_series_1w.ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1w levels to 1d timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 100  # EMA100 minimum period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_1w_aligned[i]) or 
            np.isnan(low_1w_aligned[i]) or 
            np.isnan(ema100_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_1w_aligned[i-1]  # break above previous week high
        breakout_down = close[i] < low_1w_aligned[i-1]  # break below previous week low
        
        # Trend filter: price above 1w EMA100 (bullish bias)
        price_above_ema = close[i] > ema100_1w_aligned[i]
        
        if position == 0:
            # Long: breakout up + price above EMA100 (bullish only)
            if breakout_up and price_above_ema:
                signals[i] = 0.30
                position = 1
        
        elif position == 1:
            # Exit long: break below previous week low (stop loss)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
    
    return signals

name = "1d_1w_100EMA_Breakout"
timeframe = "1d"
leverage = 1.0