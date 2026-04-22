#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot S3/R3 Breakout with 12-hour EMA trend and volume confirmation.
Long when price breaks above R3 (1-day) with bullish 12h EMA trend and volume spike.
Short when price breaks below S3 (1-day) with bearish 12h EMA trend and volume spike.
Exit when price returns to the Camarilla pivot level (midpoint).
Camarilla levels provide institutional support/resistance; volume confirms participation.
Works in bull/bear via trend filter and volatility-based levels.
"""

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
    
    # Load 1-day data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: S3, S2, S1, PP, R1, R2, R3
    # PP = (H + L + C) / 3
    # Range = H - L
    # S3 = C - (Range * 1.1 / 2)
    # R3 = C + (Range * 1.1 / 2)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    s3 = close_1d - (range_1d * 1.1 / 2)
    r3 = close_1d + (range_1d * 1.1 / 2)
    
    # Align to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Load 12h data for EMA trend - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(pivot_aligned[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3, bullish 12h EMA trend, volume spike
            if (close[i] > r3_aligned[i] and 
                ema_50_12h_aligned[i] > close_12h[-1] if len(close_12h) > 0 else False and  # Simplified trend check
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, bearish 12h EMA trend, volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema_50_12h_aligned[i] < close_12h[-1] if len(close_12h) > 0 else False and  # Simplified trend check
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to pivot level (mean reversion)
            if position == 1 and close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0