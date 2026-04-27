#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d EMA trend filter.
# Donchian breakouts capture momentum in both bull and bear markets.
# Volume confirms institutional participation.
# 1d EMA filter ensures we only trade in the direction of the higher timeframe trend.
# Session filter (08-20 UTC) reduces noise from low-volume periods.
# Target: 15-37 trades/year (60-150 over 4 years) with position size 0.20.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or filters not met
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above Donchian high + volume + uptrend
        if (close[i] > donch_high_aligned[i] and 
            volume_filter[i] and 
            close[i] > ema34_1d_aligned[i]):
            signals[i] = 0.20
            position = 1
        
        # Short entry: price breaks below Donchian low + volume + downtrend
        elif (close[i] < donch_low_aligned[i] and 
              volume_filter[i] and 
              close[i] < ema34_1d_aligned[i]):
            signals[i] = -0.20
            position = -1
        
        # Exit: reverse signal or trend change
        elif position == 1 and close[i] < ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        
        # Hold current position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0