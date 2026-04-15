#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and trend filter
# Uses 12h Donchian breakouts for trend capture, confirmed by 1d volume spike and 1d EMA50 trend.
# Designed to work in both bull and bear markets by only taking breakouts in the direction
# of the 1d trend (EMA50). Target: 50-150 total trades over 4 years (12-37/year).
# Timeframe: 12h, HTF: 1d

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + price above 1d EMA50
        if (close[i] > donch_high_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_1d_aligned[i] and
            close[i] > ema50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + price below 1d EMA50
        elif (close[i] < donch_low_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_1d_aligned[i] and
              close[i] < ema50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal
        elif position == 1 and close[i] < donch_low_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_12h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0