#!/usr/bin/env python3
"""
1h_4hDonchian_1dTrend_VolumeBreakout
Hypothesis: Use 4h Donchian breakout for direction, 1d EMA50 for trend filter, and volume spike for confirmation.
Enter on 1h breakout of 4h Donchian channel with volume confirmation. Exit when price crosses 4h Donchian midpoint.
Works in bull (buy breakouts) and bear (sell breakdowns) by using Donchian channels.
Target: 60-150 total trades over 4 years (15-37/year).
"""

name = "1h_4hDonchian_1dTrend_VolumeBreakout"
timeframe = "1h"
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
    
    # 4h Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high_4h = np.full(len(high_4h), np.nan)
    donchian_low_4h = np.full(len(low_4h), np.nan)
    
    if len(high_4h) >= 20:
        for i in range(19, len(high_4h)):
            donchian_high_4h[i] = np.max(high_4h[i-19:i+1])
            donchian_low_4h[i] = np.min(low_4h[i-19:i+1])
    
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_4h = (donchian_high_4h_aligned + donchian_low_4h_aligned) / 2.0
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection: current 1h volume > 2.0 x 20-period 1h volume average
    vol_ma20 = np.full(n, np.nan)
    if n >= 20:
        vol_ma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high, above 1d EMA50, with volume spike
            if (close[i] > donchian_high_4h_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low, below 1d EMA50, with volume spike
            elif (close[i] < donchian_low_4h_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below 4h Donchian midpoint
            if close[i] < donchian_mid_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above 4h Donchian midpoint
            if close[i] > donchian_mid_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals