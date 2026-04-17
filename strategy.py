#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_Volume_Spike
Strategy: Daily Donchian(20) breakout with weekly trend filter and volume spike.
Long: Close breaks above 20-day high + weekly EMA10 > EMA50 + volume > 2x average
Short: Close breaks below 20-day low + weekly EMA10 < EMA50 + volume > 2x average
Exit: Close returns to opposite Donchian band or trend reverses
Position size: 0.25
Designed to capture weekly trend continuation with volatility expansion.
Timeframe: 1d
"""

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
    
    # Calculate weekly EMA10 and EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    close_series_1w = pd.Series(close_1w)
    ema10_1w = close_series_1w.ewm(span=10, adjust=False, min_periods=10).mean().values
    ema50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMAs to daily timeframe
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike filter (20-period MA on daily)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema10_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_ma20[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: weekly EMA10 > EMA50 for long, < for short
        ema10_gt_ema50 = ema10_1w_aligned[i] > ema50_1w_aligned[i]
        ema10_lt_ema50 = ema10_1w_aligned[i] < ema50_1w_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: Close breaks above Donchian high + trend up + volume spike
            if (close[i] > donchian_high[i] and ema10_gt_ema50 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low + trend down + volume spike
            elif (close[i] < donchian_low[i] and ema10_lt_ema50 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close returns to Donchian low or trend reverses
            if close[i] < donchian_low[i] or not ema10_gt_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close returns to Donchian high or trend reverses
            if close[i] > donchian_high[i] or not ema10_lt_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_Spike"
timeframe = "1d"
leverage = 1.0