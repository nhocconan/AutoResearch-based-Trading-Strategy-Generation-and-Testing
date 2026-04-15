#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Uses 20-period Donchian channels for breakout signals, filtered by 12h EMA trend
# and volume spikes to avoid false breakouts. Works in both bull and bear markets
# by trading breakouts in the direction of the higher timeframe trend.
# Target: 20-50 trades per year (~80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channels on 4h
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period EMA on 12h for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate volume spike (2x 20-period median volume)
    volume_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_median[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume confirmation + price above 12h EMA
        if (close[i] > donchian_high_aligned[i] and
            volume[i] > 2.0 * volume_median[i] and
            close[i] > ema_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume confirmation + price below 12h EMA
        elif (close[i] < donchian_low_aligned[i] and
              volume[i] > 2.0 * volume_median[i] and
              close[i] < ema_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout
        elif position == 1 and close[i] < donchian_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_high_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0