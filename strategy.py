#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with daily volume confirmation and weekly trend filter.
# Donchian breakouts capture momentum in both bull and bear markets.
# Volume confirmation ensures breakouts have institutional participation.
# Weekly trend filter aligns with higher timeframe direction to avoid counter-trend trades.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate average volume (20-period) for daily confirmation
    avg_volume_1d = np.full(len(df_1d), np.nan)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    for i in range(20, len(df_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align daily average volume to 12h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate weekly EMA trend filter
    ema_1w = np.zeros(len(close_1d))
    ema_multiplier = 2 / (21 + 1)
    ema_1w[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_1w[i] = (close_1d[i] - ema_1w[i-1]) * ema_multiplier + ema_1w[i-1]
    
    # Align weekly EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_1d_aligned[i]
        weekly_ema = ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average daily volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian high with volume + above weekly EMA
            if (price > donchian_high[i] and 
                volume_confirm and
                price > weekly_ema):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume + below weekly EMA
            elif (price < donchian_low[i] and
                  volume_confirm and
                  price < weekly_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below Donchian low
            if price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above Donchian high
            if price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_Donchian_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0