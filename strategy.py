#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike
# Donchian breakouts capture momentum bursts; 12h EMA filter ensures alignment with higher timeframe trend;
# volume spike confirms institutional participation. Designed for low trade frequency to avoid fee drag.
# Works in bull markets (breakouts) and bear markets (breakdowns with trend filter).
name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12-period EMA for Donchian exit
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h EMA50 to 4h
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema12[i]) or 
            np.isnan(ema50_12h_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: price breaks above Donchian high + above 12h EMA50 + volume spike
            if close[i] > high_max[i] and close[i] > ema50_12h_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below 12h EMA50 + volume spike
            elif close[i] < low_min[i] and close[i] < ema50_12h_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA12 OR breaks below Donchian low
            if close[i] < ema12[i] or close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA12 OR breaks above Donchian high
            if close[i] > ema12[i] or close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals