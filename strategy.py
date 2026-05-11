#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation.
In bull markets, breakouts above R3 continue upward; in bear markets, breakdowns below S3 continue downward.
Uses 1d EMA34 for trend and volume > 1.5x 20-period average for confirmation.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "6h"
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
    
    # === 1D Data for Camarilla Pivots and Trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # Use previous day's values (shifted by 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day will have invalid values (from roll), but we'll handle with min_periods logic
    
    # Calculate Camarilla levels
    H_minus_L = prev_high_1d - prev_low_1d
    R3 = prev_close_1d + (H_minus_L * 1.1 / 4)
    S3 = prev_close_1d - (H_minus_L * 1.1 / 4)
    
    # Align Camarilla levels to 6h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)  # 1.5x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(100, 34)  # need EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and price > EMA34 (uptrend)
            if close[i] > R3_aligned[i] and volume_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and price < EMA34 (downtrend)
            elif close[i] < S3_aligned[i] and volume_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal signal)
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above R3 (reversal signal)
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals