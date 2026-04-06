#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6f timeframe with 1w/1d filters for directional bias.
# Uses 1w Camarilla pivot levels + volume confirmation (2x average).
# Fades at R3/S3 levels (mean reversion in extremes), breaks out at R4/S4 (momentum).
# Weekly pivot provides structural levels, volume confirms participation.
# Target: 80-150 total trades over 4 years (20-38/year) to balance signal quality and fee drag.
# Works in range via mean reversion at extremes, in trends via breakout continuation.

name = "6h_camarilla1w_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w Camarilla pivot levels (based on prior week)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point and ranges
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R4, R3, S3, S4
    r4 = pivot_1w + (range_1w * 1.1)
    r3 = pivot_1w + (range_1w * 1.1/2)
    s3 = pivot_1w - (range_1w * 1.1/2)
    s4 = pivot_1w - (range_1w * 1.1)
    
    # Align to 6t timeframe (shifted by 1 week for completed bars only)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price reaches R3 (take profit) or breaks below S3 (stop)
            if close[i] >= r3_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches S3 (take profit) or breaks above R3 (stop)
            if close[i] <= s3_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla extremes + volume
            if volume[i] > volume_threshold[i]:
                # Fade at S4/R4 (extreme levels) - mean reversion
                if close[i] <= s4_aligned[i]:
                    # Extreme low, look for bounce
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= r4_aligned[i]:
                    # Extreme high, look for pullback
                    signals[i] = -0.25
                    position = -1
                # Breakout continuation at R4/S4 with volume
                elif close[i] > r4_aligned[i]:
                    # Break above R4 with volume - continuation
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s4_aligned[i]:
                    # Break below S4 with volume - continuation
                    signals[i] = -0.25
                    position = -1
    
    return signals