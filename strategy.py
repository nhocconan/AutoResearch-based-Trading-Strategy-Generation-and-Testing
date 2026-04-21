#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTFTrend_ATRFilter_V1
Hypothesis: 4h Donchian(20) breakouts with volume spike (>1.5x 20-period volume MA) and 1d HTF trend filter (close > EMA50 for longs, < for shorts). 
Donchian channels provide objective breakout levels. Volume spikes confirm institutional participation. 
1d EMA50 ensures alignment with daily trend to avoid counter-trend trades. Target 20-50 trades/year (80-200 total over 4 years).
Uses 4h primary timeframe with 1d HTF for EMA trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike + uptrend (close > EMA50)
            if price > highest_high[i] and vol_ok and close_4h[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume spike + downtrend (close < EMA50)
            elif price < lowest_low[i] and vol_ok and close_4h[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower or trend fails
            if price < lowest_low[i] or close_4h[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper or trend fails
            if price > highest_high[i] or close_4h[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTFTrend_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0