#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA for trend filter (1d)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34 = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get weekly data for higher timeframe bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 20-period EMA for weekly trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Multi-timeframe trend filter: both 1d and 1w trend must agree
        trend_up = close[i] > ema34[i] and close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema34[i] and close[i] < ema20_1w_aligned[i]
        
        # Long conditions: price breaks above upper Donchian + trend up + volume spike
        long_breakout = (close[i] > highest_high[i-1] and trend_up and volume_spike[i])
        # Short conditions: price breaks below lower Donchian + trend down + volume spike
        short_breakout = (close[i] < lowest_low[i-1] and trend_down and volume_spike[i])
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout with volume
        elif position == 1 and close[i] < lowest_low[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i-1] and volume_spike[i]:
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

name = "6h_Donchian20_Breakout_VolumeSpike_DualTrendFilter_1d_1w"
timeframe = "6h"
leverage = 1.0