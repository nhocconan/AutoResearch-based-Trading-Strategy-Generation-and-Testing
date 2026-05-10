#!/usr/bin/env python3
"""
6h_ElderRay_Signal_1dTrend_Volume
Hypothesis: Elder Ray (Bull/Bear Power) combined with 1d EMA13 trend filter and volume confirmation.
Elder Ray measures bull/bear power relative to EMA13, providing early trend strength signals.
In trending markets, strong bull/bear power persists; in ranging markets, it fades.
Volume confirmation filters weak breakouts. Works in both bull (strong bull power) and bear (strong bear power).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_ElderRay_Signal_1dTrend_Volume"
timeframe = "6h"
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
    
    # 1d EMA13 for trend filter (Elder Ray uses EMA13)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema13_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        # Calculate EMA13 with proper initialization
        ema13_1d[12] = np.mean(close_1d[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, len(close_1d)):
            ema13_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema13_1d[i-1]
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Calculate 6-period EMA for Elder Ray (on 6h data)
    ema6 = np.full(n, np.nan)
    if n >= 6:
        ema6[5] = np.mean(close[:6])
        alpha6 = 2 / (6 + 1)
        for i in range(6, n):
            ema6[i] = alpha6 * close[i] + (1 - alpha6) * ema6[i-1]
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Using 1d EMA13 aligned to 6h timeframe
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # warmup for EMA calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(ema6[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average 1d volume (scaled to 6h)
        # Approximate 6h volume from 1d: 1d volume / 4 (since 24h/6h = 4)
        vol_6h_approx = vol_sma20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.3 * vol_6h_approx
        
        if position == 0:
            # Long: Strong bull power (> 0) with uptrend and volume confirmation
            if bull_power[i] > 0 and close[i] > ema13_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power (< 0) with downtrend and volume confirmation
            elif bear_power[i] < 0 and close[i] < ema13_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bear power becomes negative (bull power fading) or trend reversal
            if bear_power[i] < 0 or close[i] < ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bull power becomes positive (bear power fading) or trend reversal
            if bull_power[i] > 0 or close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals