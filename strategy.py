#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
- Camarilla levels calculated from prior 1w bar's high-low-close
- Long: Close breaks above R3 (strong breakout) + price > 1w EMA50 (uptrend) + volume > 2.0x 20-period avg
- Short: Close breaks below S3 (strong breakdown) + price < 1w EMA50 (downtrend) + volume > 2.0x 20-period avg
- Exit: Close reverts to Camarilla pivot point OR opposite breakout
- 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- High volume threshold (2.0x) and R3/S3 levels reduce trade frequency to avoid fee drag
- Designed for 1d timeframe to capture multi-day moves in both bull and bear markets
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
    
    # Volume confirmation: > 2.0x 20-period average (strict spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for Camarilla calculation and EMA50
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1w bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = Pivot + Range * 1.1/4, S3 = Pivot - Range * 1.1/4
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1 / 4.0
    s3_1w = pivot_1w - range_1w * 1.1 / 4.0
    pp_1w = pivot_1w  # Pivot point for mean reversion exit
    
    # Align Camarilla levels to 1d timeframe (available after 1w bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R3 + price > 1w EMA50 (uptrend) + volume spike
            if volume_spike and close[i] > r3_aligned[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + price < 1w EMA50 (downtrend) + volume spike
            elif volume_spike and close[i] < s3_aligned[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close returns to pivot point (mean reversion) OR breaks below S3 (reversal)
            if close[i] <= pp_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close returns to pivot point (mean reversion) OR breaks above R3 (reversal)
            if close[i] >= pp_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0