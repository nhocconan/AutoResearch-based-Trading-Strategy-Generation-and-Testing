#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
- Camarilla levels calculated from prior 12h bar's high-low-close
- Long: Close breaks above R3 (high volatility expansion) + price > 12h EMA50 (uptrend) + volume > 1.5x 20-period avg
- Short: Close breaks below S3 (high volatility expansion) + price < 12h EMA50 (downtrend) + volume > 1.5x 20-period avg
- Exit: Close reverts to mean (returns to Camarilla pivot point) OR opposite breakout
- 12h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe
- Works in both bull (trend continuation via breakouts) and bear (mean reversion via pivot returns)
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data ONCE before loop for Camarilla calculation and EMA50
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = Pivot + Range * 1.1/2, S3 = Pivot - Range * 1.1/2
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r3_12h = pivot_12h + range_12h * 1.1 / 2.0
    s3_12h = pivot_12h - range_12h * 1.1 / 2.0
    pp_12h = pivot_12h  # Pivot point for mean reversion exit
    
    # Align Camarilla levels to 6h timeframe (available after 12h bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R3 + price > 12h EMA50 (uptrend) + volume spike
            if volume_spike and close[i] > r3_aligned[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + price < 12h EMA50 (downtrend) + volume spike
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

name = "6h_Camarilla_R3S3_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0