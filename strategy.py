#!/usr/bin/env python3
# 12h_1d_Donchian20_Breakout_Volume_Trend
# Hypothesis: On 12h timeframe, trade breakouts from 1d-derived Donchian(20) channels with volume confirmation and 1d EMA trend filter.
# Uses 1d EMA50 to filter trades in trending markets. Targets 15-30 trades per year.
# Breakouts are confirmed by volume > 1.8x 20-period average and price beyond Donchian bands.
# Exits on reversal below/above opposite Donchian band or trend flip (price crosses 1d EMA50).
# Designed to work in both bull and bear markets by aligning with the 1d trend.

name = "12h_1d_Donchian20_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower bands (20-period)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Donchian upper, volume spike, and price above 1d EMA50 (uptrend)
            if (close[i] > donchian_upper_aligned[i] and 
                volume[i] > 1.8 * volume_ma[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below Donchian lower, volume spike, and price below 1d EMA50 (downtrend)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume[i] > 1.8 * volume_ma[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below Donchian lower or trend reversal (below EMA50)
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above Donchian upper or trend reversal (above EMA50)
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals