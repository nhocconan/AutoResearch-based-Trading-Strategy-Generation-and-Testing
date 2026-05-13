#!/usr/bin/env python3
"""
6h_Pivot_Reversal_With_Volume_Spike
Hypothesis: Price reverses at key intraday pivot levels (S1/S3, R1/R3) with volume confirmation.
Uses classic Camarilla pivot calculation from previous 12h period (appropriate for 6h timeframe).
Long at S1/S3 with volume spike, short at R1/R3 with volume spike.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee flood.
Works in both bull and bear regimes by capturing mean reversion at institutional levels.
"""

name = "6h_Pivot_Reversal_With_Volume_Spike"
timeframe = "6h"
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
    
    # Get 12h data for pivot calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # But we only need S1, S3, R1, R3
    c = df_12h['close'].values
    h = df_12h['high'].values
    l = df_12h['low'].values
    
    # Calculate pivot levels
    range_hl = h - l
    r1 = c + (range_hl * 1.1 / 12)
    r3 = c + (range_hl * 1.1 / 4)
    s1 = c - (range_hl * 1.1 / 12)
    s3 = c - (range_hl * 1.1 / 4)
    
    # Align pivot levels to 6h timeframe (previous 12h bar's levels)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price at S1 or S3 with volume spike
            if volume_spike[i] and (close[i] <= s1_aligned[i] or close[i] <= s3_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R1 or R3 with volume spike
            elif volume_spike[i] and (close[i] >= r1_aligned[i] or close[i] >= r3_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches midpoint (contrarian exit) or opposite pivot
            midpoint = (s1_aligned[i] + r1_aligned[i]) / 2
            if close[i] >= midpoint or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches midpoint or opposite pivot
            midpoint = (s1_aligned[i] + r1_aligned[i]) / 2
            if close[i] <= midpoint or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals