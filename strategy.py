#!/usr/bin/env python3
name = "12H_Donchian20_1DTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA34 for trend filter
    if len(close_1d) >= 34:
        ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema34_1d = np.full_like(close_1d, np.nan)
    
    # Align daily EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channel (20-period) on daily data
    # Using previous 20 days to avoid look-ahead
    donchian_high = np.full_like(close_1d, np.nan)
    donchian_low = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 20:  # Need 20 days of data
            donchian_high[i] = np.max(high_1d[i-20:i])
            donchian_low[i] = np.min(low_1d[i-20:i])
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 12h volume for confirmation
    volume_12h = volume.copy()
    
    # Calculate 12h volume EMA20
    if len(volume_12h) >= 20:
        vol_ema20_12h = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        vol_ema20_12h = np.full_like(volume_12h, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(34, 20)  # Need EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ema20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above daily EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        # Downtrend: price below daily EMA34
        downtrend = close[i] < ema34_1d_aligned[i]
        # Volume surge: current volume > 1.5x 12h volume EMA20
        volume_surge = volume[i] > vol_ema20_12h[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price breaks above Donchian high + volume surge
            if uptrend and close[i] > donchian_high_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below Donchian low + volume surge
            elif downtrend and close[i] < donchian_low_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below Donchian low
            if not uptrend or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above Donchian high
            if not downtrend or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals