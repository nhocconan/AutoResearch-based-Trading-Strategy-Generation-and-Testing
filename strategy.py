#!/usr/bin/env python3
"""
1d Daily Range Breakout with Weekly Trend Filter
Hypothesis: Breakouts from the previous day's high/low range capture momentum,
while the weekly trend filter ensures alignment with higher-timeframe direction.
This strategy targets 15-25 trades per year (~60-100 total over 4 years) to minimize
fee drag while capturing meaningful moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_daily_range_breakout_weekly_trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Previous day's high and low (for breakout)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Weekly EMA50 Trend Filter
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below previous day's low
            if close[i] < prev_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above previous day's high
            if close[i] > prev_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above previous day's high + price above weekly EMA50
            if (close[i] > prev_high[i] and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below previous day's low + price below weekly EMA50
            elif (close[i] < prev_low[i] and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals