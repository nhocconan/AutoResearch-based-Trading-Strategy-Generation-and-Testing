#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Donchian20_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike detection (15-period average)
    vol_avg = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema20_1d[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 15-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above 20-day high with weekly uptrend and volume spike
            if close[i] > high_max[i] and close[i] > ema20_1d[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-day low with weekly downtrend and volume spike
            elif close[i] < low_min[i] and close[i] < ema20_1d[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below 20-day low OR weekly trend turns down
            if close[i] < low_min[i] or close[i] < ema20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above 20-day high OR weekly trend turns up
            if close[i] > high_max[i] or close[i] > ema20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals