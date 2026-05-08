#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Breakout_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian(20) for trend direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Daily Donchian(10) for entry
    upper_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lower_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Daily volume spike: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(upper_10[i]) or np.isnan(lower_10[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_20_val = upper_20_aligned[i]
        lower_20_val = lower_20_aligned[i]
        upper_10_val = upper_10[i]
        lower_10_val = lower_10[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above weekly upper + daily breakout + volume spike
            if (close[i] > upper_20_val and 
                close[i] > upper_10_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly lower + daily breakdown + volume spike
            elif (close[i] < lower_20_val and 
                  close[i] < lower_10_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly lower OR daily lower
            if (close[i] < lower_20_val or close[i] < lower_10_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly upper OR daily upper
            if (close[i] > upper_20_val or close[i] > upper_10_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals