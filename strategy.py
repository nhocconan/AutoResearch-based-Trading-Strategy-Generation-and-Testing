#!/usr/bin/env python3
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
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get daily data for entry signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 50-period EMA on weekly for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period Donchian channels on daily
    upper_channel_1d = np.full_like(close_1d, np.nan)
    lower_channel_1d = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        upper_channel_1d[i] = np.max(high_1d[i-19:i+1])
        lower_channel_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-period SMA on daily for volume filter
    sma_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly EMA to daily
    ema_50_1w_aligned_daily = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align daily indicators to 1d (no alignment needed as we're using 1d timeframe)
    # But we need to align to the 1d timeframe properly (already aligned by construction)
    # For safety, we'll align but it should be identity for same timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel_1d)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel_1d)
    sma_vol_aligned = align_htf_to_ltf(prices, df_1d, sma_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(ema_50_1w_aligned_daily[i]) or np.isnan(sma_vol_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly EMA
        uptrend = close[i] > ema_50_1w_aligned_daily[i]
        downtrend = close[i] < ema_50_1w_aligned_daily[i]
        
        # Volume filter: current volume above 20-day average
        volume_filter = volume[i] > sma_vol_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend and volume
            if close[i] > upper_channel_aligned[i] and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend and volume
            elif close[i] < lower_channel_aligned[i] and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian channel OR trend reverses
            if (close[i] < lower_channel_aligned[i]) or (not uptrend):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian channel OR trend reverses
            if (close[i] > upper_channel_aligned[i]) or (not downtrend):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0