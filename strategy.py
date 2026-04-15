#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and 1-day ATR filter
# Designed for low trade frequency (target 20-40/year) with clear trend-following logic
# Works in both bull (breakout above upper band) and bear (breakdown below lower band) markets
# Uses Donchian channels from 4h, volume spike from 1d to confirm interest, and ATR for volatility filtering

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Donchian and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    # Using previous period's data to avoid look-ahead
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR (14-period) on 4h for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[np.nan], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[np.nan], close_4h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + ATR filter
        if (high[i] > donchian_high_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            atr_aligned[i] > 0 and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + ATR filter
        elif (low[i] < donchian_low_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              atr_aligned[i] > 0 and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to midpoint of Donchian channel
        elif position == 1 and close[i] < (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dVolume_ATR_Filter"
timeframe = "4h"
leverage = 1.0