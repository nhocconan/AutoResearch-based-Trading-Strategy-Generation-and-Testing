#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume spike (>1.8x 20-period median).
Uses 1w HTF for trend alignment to capture major moves while avoiding whipsaws. Discrete sizing (0.25) minimizes fee drag.
Target: 50-100 trades over 4 years (12-25/year). Works in bull/bear via weekly trend filter + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.8x 20-period median (balanced to avoid overtrading)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.8 * vol_median)
    
    # Load weekly data for HTF trend filter (EMA50) and Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period volume median, 20-period Donchian, 50-period EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: close above weekly Donchian high + volume confirm + bullish weekly trend
        if close[i] > donchian_high[i] and volume_confirm[i] and close[i] > ema50_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: close below weekly Donchian low + volume confirm + bearish weekly trend
        elif close[i] < donchian_low[i] and volume_confirm[i] and close[i] < ema50_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits when price touches weekly Donchian low, short exits when price touches weekly Donchian high
        elif position == 1 and close[i] <= donchian_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= donchian_high[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0