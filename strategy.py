#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Donchian channels provide robust breakout signals. Weekly trend filter ensures
# alignment with higher timeframe momentum. Volume spike confirms breakout strength.
# Designed for 1d timeframe targeting 30-100 trades over 4 years (7-25/year).
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (using 1d as base for 1d timeframe)
    # Actually, for 1d timeframe, we can use the prices directly for Donchian
    # But we need 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) on daily data
    # Highest high of last 20 days
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 days
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.8x 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Price breaks above Donchian upper band + weekly uptrend + volume
        if (close[i] > highest_high[i-1] and  # Break above previous period's high
            close[i] > ema50_1w_aligned[i] and  # Price above weekly EMA50 (uptrend)
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Price breaks below Donchian lower band + weekly downtrend + volume
        elif (close[i] < lowest_low[i-1] and   # Break below previous period's low
              close[i] < ema50_1w_aligned[i] and  # Price below weekly EMA50 (downtrend)
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0