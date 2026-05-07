#!/usr/bin/env python3
# 12h_1d_Camarilla_R2S2_Breakout_VolumeSpike
# Hypothesis: 12h timeframe with 1d Camarilla R2/S2 breakout and volume spike (3x) reduces trade frequency to optimal levels (12-37/year) while maintaining edge in bull/bear markets via price action at institutional levels. Uses tighter exits at R3/S3 to limit drawdown.

name = "12h_1d_Camarilla_R2S2_Breakout_VolumeSpike"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate all Camarilla levels
    hl_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * hl_range / 2
    r2_1d = close_1d + 1.1 * hl_range / 6
    r1_1d = close_1d + 1.1 * hl_range / 12
    s1_1d = close_1d - 1.1 * hl_range / 12
    s2_1d = close_1d - 1.1 * hl_range / 6
    s3_1d = close_1d - 1.1 * hl_range / 2
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align all levels to 12h timeframe (use previous day's levels)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Volume spike detection: 3.0x average volume (50-period for stability)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R2 with volume spike (>3x)
            if close[i] > r2_1d_aligned[i] and volume[i] > 3.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume spike (>3x)
            elif close[i] < s2_1d_aligned[i] and volume[i] > 3.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below R3 (tighter exit)
            if close[i] <= r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above S3 (tighter exit)
            if close[i] >= s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals