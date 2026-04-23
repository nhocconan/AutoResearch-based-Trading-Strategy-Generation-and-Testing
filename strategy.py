#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d Camarilla Pivot with Volume Confirmation
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (momentum)
- 1d Camarilla: R3/S3 as mean reversion levels, R4/S4 as breakout levels
- Long: Bull Power > 0 + price <= S3 + volume > 1.5x 20-period avg (fade at support)
- Short: Bear Power < 0 + price >= R3 + volume > 1.5x 20-period avg (fade at resistance)
- Exit: Opposite power crosses zero or price crosses EMA13
- Uses Elder Ray for momentum, Camarilla for key levels, volume for confirmation
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in ranging markets (fade at R3/S3) and can capture breaks at R4/S4
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.0 * range_1d
    s3 = close_1d - 1.0 * range_1d
    r4 = close_1d + 1.5 * range_1d
    s4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Elder Ray Power (using 13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13)  # Need 20 for volume MA, 13 for EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 + price at or below S3 (fade at support) + volume
            # Optional: break above R4 for continuation
            if (bull_power[i] > 0 and 
                volume_confirm and
                (close[i] <= s3_aligned[i] or close[i] >= r4_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 + price at or above R3 (fade at resistance) + volume
            # Optional: break below S4 for continuation
            elif (bear_power[i] < 0 and 
                  volume_confirm and
                  (close[i] >= r3_aligned[i] or close[i] <= s4_aligned[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR price crosses below EMA13 (momentum loss)
            if bull_power[i] <= 0 or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR price crosses above EMA13 (momentum loss)
            if bear_power[i] >= 0 or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Camarilla_R3S3_VolumeConfirm"
timeframe = "6h"
leverage = 1.0