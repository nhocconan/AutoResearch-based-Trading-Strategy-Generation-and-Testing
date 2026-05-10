#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
Hypothesis: Use Camarilla pivot levels from daily timeframe for precise entry/exit.
In trending markets (determined by 12h EMA50), price tends to respect Camarilla levels.
Long when price breaks above R1 with volume confirmation in uptrend.
Short when price breaks below S1 with volume confirmation in downtrend.
Exit when price reaches opposite S1/R1 level or trend reverses.
Targets 150-250 trades over 4 years (38-63/year) to balance opportunity and cost.
Works in both bull (buy breakouts) and bear (sell breakdowns).
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h volume SMA20 for volume confirmation
    volume_12h = df_12h['volume'].values
    vol_sma20_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        vol_sma20_12h[19] = np.mean(volume_12h[:20])
        for i in range(20, len(volume_12h)):
            vol_sma20_12h[i] = (vol_sma20_12h[i-1] * 19 + volume_12h[i]) / 20
    vol_sma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma20_12h)
    
    # Daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    # P = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + camarilla_range * 1.1 / 12
    s1_1d = close_1d - camarilla_range * 1.1 / 12
    r4_1d = close_1d + camarilla_range * 1.1 / 2
    s4_1d = close_1d - camarilla_range * 1.1 / 2
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1)  # Need EMA50 and at least one Camarilla level
    
    for i in range(start_idx, n):
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_12h_aligned[i]) or \
           np.isnan(camarilla_pivot_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or \
           np.isnan(s4_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 12h volume (scaled to 4h)
        vol_4h_approx = vol_sma20_12h_aligned[i] / 3.0  # 3x 4h periods in 12h
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation in uptrend
            if close[i] > r1_1d_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation in downtrend
            elif close[i] < s1_1d_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price reaches S1 level or trend reversal
            if close[i] < s1_1d_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price reaches R1 level or trend reversal
            if close[i] > r1_1d_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals