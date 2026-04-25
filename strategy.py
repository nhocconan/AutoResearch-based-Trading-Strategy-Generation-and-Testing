#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 6h timeframe using 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) with 1d EMA50 trend filter and daily volume spike (>2.0x 20-bar MA) for confirmation.
Enter long when price breaks above R4 AND above EMA50 AND volume spike. Enter short when price breaks below S4 AND below EMA50 AND volume spike.
Exit on touch of opposite Camarilla level (R3 for longs, S3 for shorts) or trend reversal.
Uses discrete sizing 0.25 to manage drawdown. Target 12-37 trades/year on 6h timeframe.
Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.
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
    
    # Get 1d data for Camarilla pivots, EMA50, and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla pivot levels
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # R4 = PP + Range * 1.1/2, R3 = PP + Range * 1.1/4
    # S3 = PP - Range * 1.1/4, S4 = PP - Range * 1.1/2
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = pp_1d + (range_1d * 1.1 / 2)
    r3_1d = pp_1d + (range_1d * 1.1 / 4)
    s3_1d = pp_1d - (range_1d * 1.1 / 4)
    s4_1d = pp_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (completed daily bar only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R4 AND above EMA50 AND volume spike
            long_setup = (close[i] > r4_1d_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price breaks below S4 AND below EMA50 AND volume spike
            short_setup = (close[i] < s4_1d_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches R3 (fade level) OR closes below EMA50
            if (close[i] <= r3_1d_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches S3 (fade level) OR closes above EMA50
            if (close[i] >= s3_1d_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0