#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Price breaks Camarilla R1 (long) or S1 (short) from prior day, with 12h EMA50 trend filter, volume spike confirmation, and exit on reversion.
Designed to work in both bull and bear markets by trading only in direction of 12h trend.
Targets 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
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
    
    # 12h data for HTF trend and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_12h = df_1d['low'].values  # Reuse variable for clarity
    close_1d = df_1d['close'].values
    
    # Camarilla levels from prior day: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_12h) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_12h) / 12
    
    # 12h EMA50 for trend filter
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    
    # 12h volume SMA20 for volume spike detection
    vol_sma20_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        vol_sma20_12h[19] = np.mean(volume_12h[:20])
        for i in range(20, len(volume_12h)):
            vol_sma20_12h[i] = (vol_sma20_12h[i-1] * 19 + volume_12h[i]) / 20
    
    # Align 1d indicators to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Align 12h indicators to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_sma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 2x average 12h volume (scaled)
        # 3x 4h bars in 12h, so scale 12h average by 3 for per-4h-bar comparison
        vol_12h_scaled = vol_sma20_12h_aligned[i] / 3.0
        volume_spike = volume[i] > 2.0 * vol_12h_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema50_12h_aligned[i]
        is_downtrend = close[i] < ema50_12h_aligned[i]
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, in uptrend, with volume spike
            if price_above_r1 and is_uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, in downtrend, with volume spike
            elif price_below_s1 and is_downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R1 or trend turns down
            if not price_above_r1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S1 or trend turns up
            if not price_below_s1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals