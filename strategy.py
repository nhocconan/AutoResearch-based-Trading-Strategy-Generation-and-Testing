#!/usr/bin/env python3
# 4h_1d_Pivot_R2S2_Breakout_Volume
# Hypothesis: Trade momentum breakouts from 1d R2/S2 levels on 4h timeframe with volume confirmation.
# Uses 1-day pivot points (R2/S2) as dynamic support/resistance, requiring price to break these levels with elevated volume.
# Designed for 20-50 trades per year by requiring both price break and volume surge.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by trading in direction of breakout.

name = "4h_1d_Pivot_R2S2_Breakout_Volume"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day pivot points and R2/S2 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Daily range
    range_1d = high_1d - low_1d
    
    # Camarilla-style R2 and S2 levels (commonly used breakout levels)
    # R2 = Close + 1.1 * Range / 6
    # S2 = Close - 1.1 * Range / 6
    r2_1d = close_1d + (1.1 * range_1d) / 6.0
    s2_1d = close_1d - (1.1 * range_1d) / 6.0
    
    # Align 1d levels to 4h timeframe (waits for 1d bar to close)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R2 with volume surge (2x average)
            if close[i] > r2_aligned[i] and volume[i] > 2.0 * volume_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume surge (2x average)
            elif close[i] < s2_aligned[i] and volume[i] > 2.0 * volume_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: hold until price breaks back below S2 (reversal signal)
            if close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: hold until price breaks back above R2 (reversal signal)
            if close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals