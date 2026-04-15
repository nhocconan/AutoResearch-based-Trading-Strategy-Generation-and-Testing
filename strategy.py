#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter
# Designed for moderate trade frequency (target 20-40/year) with clear trend-following logic
# Works in both bull (breakout above upper band in uptrend) and bear (breakdown below lower band in downtrend) markets
# Uses Donchian channels from 4h, volume spike to confirm breakout strength, and weekly EMA for trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 4h - using previous period to avoid look-ahead
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to lower timeframe (15m for precision)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size as fraction of capital
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + uptrend + volume spike
        if (high[i] > donchian_high_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian low + downtrend + volume spike
        elif (low[i] < donchian_low_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or price returns to middle of Donchian channel
        elif position == 1 and close[i] < (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dVolume_1wEMA_Trend"
timeframe = "4h"
leverage = 1.0