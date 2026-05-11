# 12h_Camarilla_Pivot_Support_Resistance_Breakout
# Hypothesis: Use daily Camarilla pivot levels on 12h timeframe with volume confirmation.
# Camarilla levels act as natural support/resistance; breakouts with volume indicate momentum.
# Works in both bull/bear markets as it follows breakout direction. Low trade frequency target.
# Uses 1-day Camarilla levels (S1,S2,S3,R1,R2,R3) for breakout detection.

#!/usr/bin/env python3
name = "12h_Camarilla_Pivot_Support_Resistance_Breakout"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500), R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833), S1 = C - ((H-L) * 1.0833), S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500), S4 = C - ((H-L) * 1.5000)
    # We'll use R1,S1,R2,S2,R3,S3 for breakout signals
    
    # Previous day's OHLC (using shift(1) to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.0833)
    r2 = prev_close + (rng * 1.1666)
    r3 = prev_close + (rng * 1.2500)
    s1 = prev_close - (rng * 1.0833)
    s2 = prev_close - (rng * 1.1666)
    s3 = prev_close - (rng * 1.2500)
    
    # Align daily levels to 12h timeframe (already delayed by shift(1) above)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume
            if close[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal) or volume dries up
            if close[i] < s1_aligned[i] or volume[i] < vol_ma20[i] * 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 (reversal) or volume dries up
            if close[i] > r1_aligned[i] or volume[i] < vol_ma20[i] * 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals