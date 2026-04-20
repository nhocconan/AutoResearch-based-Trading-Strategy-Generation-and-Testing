#!/usr/bin/env python3
# 4h_Donchian20_Breakout_Volume_TrendFilter
# Hypothesis: On 4h timeframe, trade breakouts from Donchian(20) channels with volume confirmation and 1h EMA50 trend filter.
# Uses 1h EMA50 to align with trend direction. Targets 20-40 trades per year. Works in bull/bear via trend-aligned breakouts.
# Volume spike (2x 20-period average) confirms breakout strength. Avoids false breakouts in low-volume environments.

name = "4h_Donchian20_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for trend filter (don't need daily to avoid look-ahead)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 1h EMA50 for trend filter
    close_1h = df_1h['close'].values
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Donchian upper, volume spike, and price above 1h EMA50 (uptrend)
            if (close[i] > high_max[i] and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below Donchian lower, volume spike, and price below 1h EMA50 (downtrend)
            elif (close[i] < low_min[i] and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below Donchian lower or trend reversal (below EMA50)
            if close[i] < low_min[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above Donchian upper or trend reversal (above EMA50)
            if close[i] > high_max[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals