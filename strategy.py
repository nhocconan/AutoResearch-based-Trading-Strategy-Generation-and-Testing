#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout (20-period) with 12h trend filter (EMA50) and volume spike confirmation
# Breakouts from Donchian channels capture strong momentum moves. The 12h EMA50 filter ensures
# trades align with the higher timeframe trend, reducing whipsaw in ranging markets.
# Volume spike confirms institutional participation. Designed for low trade frequency.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_Donchian20_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + uptrend + volume spike
            if (close[i] > upper_channel and 
                close[i] > ema50_12h_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + downtrend + volume spike
            elif (close[i] < lower_channel and 
                  close[i] < ema50_12h_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR closes below EMA50
            if close[i] < lower_channel or close[i] < ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR closes above EMA50
            if close[i] > upper_channel or close[i] > ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals