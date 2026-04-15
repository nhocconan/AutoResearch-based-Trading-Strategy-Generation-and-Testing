#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + weekly trend filter
# Uses Donchian channel breakouts for trend capture, volume to confirm breakout strength,
# and weekly EMA50 to filter trend direction. Works in both bull and bear by
# only taking breakouts in the direction of the weekly trend (EMA50).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on weekly for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + price above weekly EMA50
        if (close[i] > donch_high_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema50_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + price below weekly EMA50
        elif (close[i] < donch_low_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema50_1w_aligned[i] and
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

name = "12h_Donchian_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0