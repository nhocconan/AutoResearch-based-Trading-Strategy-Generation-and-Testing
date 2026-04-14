#!/usr/bin/env python3
"""
4h_1d_Camarilla_Volume_Confirmation_v1
Hypothesis: On 4h timeframe, use Camarilla pivot levels from daily timeframe for entries.
Buy near S1/S2 with volume > 1.5x average, sell near R1/R2 with volume > 1.5x average.
Exits on opposite touch (long exits at R1/R2, short exits at S1/S2).
Daily pivots provide stable support/resistance, volume confirms institutional interest.
Designed for 15-35 trades/year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 20-period average volume on 4h data
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    start = 20  # Need volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        volume_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        if position == 0:
            # Long entry: price near S1/S2 with volume confirmation
            if ((close[i] <= s1_aligned[i] * 1.002 or close[i] <= s2_aligned[i] * 1.002) and
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short entry: price near R1/R2 with volume confirmation
            elif ((close[i] >= r1_aligned[i] * 0.998 or close[i] >= r2_aligned[i] * 0.998) and
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R1 or R2 (take profit)
            if (close[i] >= r1_aligned[i] * 0.998 or close[i] >= r2_aligned[i] * 0.998):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches S1 or S2 (take profit)
            if (close[i] <= s1_aligned[i] * 1.002 or close[i] <= s2_aligned[i] * 1.002):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0