#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Trend Filter + Volume Spike (1h timeframe)
# Williams %R measures momentum overbought/oversold conditions.
# %R < -80 = oversold (buy signal), %R > -20 = overbought (sell signal)
# Trend filter: 1d EMA(34) - only take longs in uptrend, shorts in downtrend
# Volume spike confirms institutional participation
# Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
# Uses 4h/1d for signal direction, 1h only for entry timing
# Session filter (08-20 UTC) reduces noise trades

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    willr = -100 * (highest_high - close) / hl_range  # Williams %R
    
    # Get 4h data for Williams %R calculation (primary signal)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R on 4h
    highest_high_4h = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    hl_range_4h = highest_high_4h - lowest_low_4h
    hl_range_4h = np.where(hl_range_4h == 0, 1e-10, hl_range_4h)
    willr_4h = -100 * (highest_high_4h - close_4h) / hl_range_4h
    
    # Align 4h Williams %R to 1h
    willr_4h_aligned = align_htf_to_ltf(prices, df_4h, willr_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.8x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC (reduce noise)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(session_filter[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Williams %R signals with trend and volume filters
        if willr_4h_aligned[i] < -80:  # Oversold - potential long
            if close[i] > ema34_1d_aligned[i] and volume_filter[i]:  # Uptrend + volume
                signals[i] = 0.20
                position = 1
        elif willr_4h_aligned[i] > -20:  # Overbought - potential short
            if close[i] < ema34_1d_aligned[i] and volume_filter[i]:  # Downtrend + volume
                signals[i] = -0.20
                position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsR_1dEMA34_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0