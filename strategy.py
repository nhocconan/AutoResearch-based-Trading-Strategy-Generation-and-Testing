#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and 1d trend filter
# Uses Donchian(20) on 6h for entry, confirmed by 12h volume spike and 1d EMA trend.
# Works in bull markets (breakouts above upper band with uptrend) and bear markets
# (breakouts below lower band with downtrend). Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian (self-referential but needed for calculation)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channels (20-period) on 6h
    highest_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume MA(20) for spike detection
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_6h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_6h, lowest_20)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            continue
        
        # Volume spike condition: current 12h volume > 1.5x its 20-period MA
        vol_12h_current = volume_12h[i // 2] if i // 2 < len(volume_12h) else volume_12h[-1]
        vol_spike = vol_12h_current > 1.5 * vol_ma_20_aligned[i]
        
        # Long entry: price breaks above Donchian upper + volume spike + uptrend (price > EMA50)
        if (close[i] > highest_20_aligned[i] and
            vol_spike and
            close[i] > ema_50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian lower + volume spike + downtrend (price < EMA50)
        elif (close[i] < lowest_20_aligned[i] and
              vol_spike and
              close[i] < ema_50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Donchian breakout or loss of volume spike
        elif position == 1 and (close[i] < lowest_20_aligned[i] or not vol_spike):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_20_aligned[i] or not vol_spike):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Volume_Trend"
timeframe = "6h"
leverage = 1.0