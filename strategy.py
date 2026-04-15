#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + volume confirmation + 12h EMA50 trend filter
# Uses Donchian channel breakouts for trend capture, volume to confirm breakout strength,
# and 12h EMA50 for trend filter to avoid counter-trend trades. Works in both bull and bear
# by only taking breakouts in the direction of the 12h trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 6h
    donch_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    donch_high_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_high_6h)
    donch_low_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_low_6h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_6h_aligned[i]) or np.isnan(donch_low_6h_aligned[i]) or
            np.isnan(ema50_12h_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + price above 12h EMA50
        if (close[i] > donch_high_6h_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema50_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + price below 12h EMA50
        elif (close[i] < donch_low_6h_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema50_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and close[i] < donch_low_6h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_6h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Volume_12hEMA50_Trend"
timeframe = "6h"
leverage = 1.0