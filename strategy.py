#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
# Donchian channels (20-period) identify breakouts from price consolidation.
# 12h EMA50 filters for trend direction to avoid counter-trend trades.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Designed for ~25-35 trades/year per symbol to avoid fee drag.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above Donchian upper band + uptrend + volume
        if (close[i] > highest_high[i] and 
            close[i] > ema50_12h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian lower band + downtrend + volume
        elif (close[i] < lowest_low[i] and 
              close[i] < ema50_12h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel
        else:
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if abs(close[i] - donchian_mid) < (highest_high[i] - lowest_low[i]) * 0.1:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0