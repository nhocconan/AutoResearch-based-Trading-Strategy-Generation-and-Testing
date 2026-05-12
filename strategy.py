#!/usr/bin/env python3
"""
6h_PivotReversal_VolumeExhaustion
Hypothesis: On 6h, trade reversals at daily pivot levels (S1/S2/R1/R2) when volume shows exhaustion (current volume < 50% of 20-period average) and price shows rejection (close near open). Uses pivot levels as natural support/resistance that work in both trending and ranging markets. Low frequency (~20 trades/year) due to strict volume exhaustion filter.
"""

name = "6h_PivotReversal_VolumeExhaustion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_ = prices['open'].values
    volume = prices['volume'].values

    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate daily pivot points (standard formula)
    # P = (H + L + C) / 3
    # S1 = 2*P - H
    # S2 = P - (H - L)
    # R1 = 2*P - L
    # R2 = P + (H - L)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    s1 = 2 * pivot - high_1d
    s2 = pivot - (high_1d - low_1d)
    r1 = 2 * pivot - low_1d
    r2 = pivot + (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)

    # Volume exhaustion: current volume < 50% of 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price rejection: close near open (small body)
    body_size = np.abs(close - open_)
    candle_range = high - low
    body_ratio = np.where(candle_range > 0, body_size / candle_range, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(s2_6h[i]) or 
            np.isnan(r1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(body_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume exhaustion condition
        volume_exhausted = volume[i] < vol_avg_20[i] * 0.5
        
        # Price rejection condition (small body)
        price_rejected = body_ratio[i] < 0.3  # body less than 30% of range

        if position == 0:
            # LONG: Near S1/S2 with volume exhaustion and price rejection
            near_support = (low[i] <= s1_6h[i] * 1.002 and low[i] >= s2_6h[i] * 0.998) or \
                          (low[i] <= s2_6h[i] * 1.002 and low[i] >= s2_6h[i] * 0.998)
                          
            if near_support and volume_exhausted and price_rejected:
                signals[i] = 0.25
                position = 1
            # SHORT: Near R1/R2 with volume exhaustion and price rejection
            elif (high[i] >= r1_6h[i] * 0.998 and high[i] <= r1_6h[i] * 1.002) or \
                 (high[i] >= r2_6h[i] * 0.998 and high[i] <= r2_6h[i] * 1.002):
                if volume_exhausted and price_rejected:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves above pivot or stops showing rejection
            if close[i] > pivot_6h[i] or body_ratio[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves below pivot or stops showing rejection
            if close[i] < pivot_6h[i] or body_ratio[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

def align_htf_to_ltf(prices, df_htf, values):
    """Simple alignment function - in practice use the one from mtf_data"""
    from mtf_data import align_htf_to_ltf
    return align_htf_to_ltf(prices, df_htf, values)