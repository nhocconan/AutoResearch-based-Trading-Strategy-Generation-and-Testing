#!/usr/bin/env python3
name = "4h_ComboBreakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Donchian breakout (entry timing) ===
    # Use 1h data for entry timing precision, but trend from 4h
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    donchian_high = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1h, donchian_low)
    
    # === 4h trend filter: EMA50 ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 4h volume filter ===
    volume_4h = df_4h['volume'].values
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume_4h > (1.5 * vol_avg_4h)
    volume_surge_aligned = align_htf_to_ltf(prices, df_4h, volume_surge.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or
            np.isnan(volume_surge_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + above EMA50 + volume surge
            if (close[i] > donchian_high_aligned[i] and
                close[i] > ema50_4h_aligned[i] and
                volume_surge_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + below EMA50 + volume surge
            elif (close[i] < donchian_low_aligned[i] and
                  close[i] < ema50_4h_aligned[i] and
                  volume_surge_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Break below Donchian low or below EMA50
            if close[i] < donchian_low_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Break above Donchian high or above EMA50
            if close[i] > donchian_high_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals