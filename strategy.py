#!/usr/bin/env python3
"""
4h_1d_ElderRay_BullBear_12hVolume
Hypothesis: Combines 1-day Elder Ray Index (bull power/bear power) with 12-hour volume confirmation to capture strong momentum moves in both bull and bear markets. Uses Elder Ray > 0 for long, < 0 for short with volume > 2x 20-period average. Designed for low trade frequency (~20-30 trades/year) by requiring strong directional conviction.
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
    
    # Calculate 1-day EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # EMA13 on daily close
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Elder Ray = Bull Power + Bear Power (net strength)
    elder_ray = bull_power + bear_power
    
    # Align Elder Ray to 4h timeframe (wait for previous day's close)
    elder_ray_aligned = align_htf_to_ltf(prices, df_1d, elder_ray)
    
    # 12-hour volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(elder_ray_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        elder_ray_val = elder_ray_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: positive Elder Ray (bullish) + volume confirmation
            if elder_ray_val > 0 and vol_conf:
                signals[i] = size
                position = 1
            # Short: negative Elder Ray (bearish) + volume confirmation
            elif elder_ray_val < 0 and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Elder Ray turns negative
            if elder_ray_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Elder Ray turns positive
            if elder_ray_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_1d_ElderRay_BullBear_12hVolume"
timeframe = "4h"
leverage = 1.0