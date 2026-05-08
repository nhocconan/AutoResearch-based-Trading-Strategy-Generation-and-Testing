#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# Uses 1h for entry timing only, 4h for trend direction and 1d for regime filter
# Targets 60-150 total trades over 4 years = 15-37/year for 1h
# Designed to work in both bull and bear markets via trend alignment and volume filters

name = "1h_Donchian_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data once for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d average volume for regime filter
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 1h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        avg_vol_1d_val = avg_vol_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + 4h uptrend + volume spike
            if (high[i] > high_max[i] and 
                close[i] > ema50_4h_val and 
                volume[i] > avg_vol_1d_val and  # volume above 1d average
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below Donchian low + 4h downtrend + volume spike
            elif (low[i] < low_min[i] and 
                  close[i] < ema50_4h_val and 
                  volume[i] > avg_vol_1d_val and  # volume above 1d average
                  vol_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 4h trend turns down
            if (low[i] < low_min[i] or close[i] < ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 4h trend turns up
            if (high[i] > high_max[i] or close[i] > ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals