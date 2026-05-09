#!/usr/bin/env python3
name = "1D_1W_Triple_Confirmation"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA for weekly trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Donchian channel and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channel (previous day)
    donchian_high = np.full_like(close_1d, np.nan)
    donchian_low = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 20:
            donchian_high[i] = np.max(high_1d[i-20:i])
            donchian_low[i] = np.min(low_1d[i-20:i])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 20-period volume average (previous day)
    vol_ma20 = np.full_like(volume_1d, np.nan)
    
    for i in range(len(volume_1d)):
        if i >= 20:
            vol_ma20[i] = np.mean(volume_1d[i-20:i])
    
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_surge = volume[i] > vol_ma20_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: Weekly uptrend + price breaks above Donchian high + volume surge
            if weekly_uptrend and close[i] > donchian_high_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Weekly downtrend + price breaks below Donchian low + volume surge
            elif weekly_downtrend and close[i] < donchian_low_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly trend turns down OR price breaks below Donchian low
            if not weekly_uptrend or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly trend turns up OR price breaks above Donchian high
            if not weekly_downtrend or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals