#!/usr/bin/env python3
"""
#100738 - 1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_Volume
Hypothesis: Daily breakout strategy using weekly EMA50 trend filter and Camarilla R1/S1 levels.
Targets 20-50 trades/year to minimize fee drag. Works in bull (breakouts with trend) and bear (mean reversion to pivot).
Uses 1d primary timeframe with 1w HTF for trend filter and daily pivot calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily pivot levels (for same day)
    pivot = (high + low + close) / 3
    daily_range = high - low
    r1 = close + daily_range * 1.1 / 12
    s1 = close - daily_range * 1.1 / 12
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r1[i]) or 
            np.isnan(s1[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R1, above 1w EMA50, volume spike
        if (close[i] > r1[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S1, below 1w EMA50, volume spike
        elif (close[i] < s1[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to daily pivot (mean reversion)
        elif position == 1 and close[i] < pivot[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > pivot[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0