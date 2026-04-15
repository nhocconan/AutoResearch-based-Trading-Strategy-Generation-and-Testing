#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Uses 6h Donchian breakouts for entry, 1-week EMA50 for trend filter (only long in weekly uptrend, short in downtrend),
# and volume confirmation to avoid false breakouts. Designed to work in both bull and bear by
# following the weekly trend direction. Target: 60-120 total trades over 4 years (15-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 6h
    donch_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on weekly for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 6h)
    vol_avg_6h = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    donch_high_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_high_6h)
    donch_low_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_low_6h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_6h, vol_avg_6h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_6h_aligned[i]) or np.isnan(donch_low_6h_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above 6h Donchian high + volume spike + weekly uptrend
        if (close[i] > donch_high_6h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            close[i] > ema50_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below 6h Donchian low + volume spike + weekly downtrend
        elif (close[i] < donch_low_6h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              close[i] < ema50_1w_aligned[i] and
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

name = "6h_Donchian_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0