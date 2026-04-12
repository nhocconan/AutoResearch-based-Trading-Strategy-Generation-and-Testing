#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Reversal_v1
Hypothesis: Use daily Camarilla pivot levels with mean reversion on 12h timeframe.
Long when price touches or breaks below S3 with reversal confirmation, short when touches or breaks above R3.
Camarilla levels work well in ranging markets (common in 2025-2026) and provide clear reversal zones.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drift.
Works in bull via buying dips to support, in bear via selling rallies to resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate Camarilla pivot levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Camarilla levels: R3, R4, S3, S4
    camarilla_r3 = prev_close + range_val * 1.1 / 2
    camarilla_r4 = prev_close + range_val * 1.1
    camarilla_s3 = prev_close - range_val * 1.1 / 2
    camarilla_s4 = prev_close - range_val * 1.1
    
    # Align daily levels to 12h timeframe
    camarilla_r3_array = np.full(len(df_1d), camarilla_r3)
    camarilla_r4_array = np.full(len(df_1d), camarilla_r4)
    camarilla_s3_array = np.full(len(df_1d), camarilla_s3)
    camarilla_s4_array = np.full(len(df_1d), camarilla_s4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_array)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_array)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_array)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_array)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion conditions with volume filter
        long_setup = low[i] <= camarilla_s3_aligned[i] and vol_ratio[i] > 1.3
        short_setup = high[i] >= camarilla_r3_aligned[i] and vol_ratio[i] > 1.3
        
        # Exit conditions: return to daily pivot (mean reversion complete)
        camarilla_pivot = (prev_high + prev_low + 2 * prev_close) / 4
        camarilla_pivot_array = np.full(len(df_1d), camarilla_pivot)
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_array)
        
        long_exit = close[i] >= camarilla_pivot_aligned[i]
        short_exit = close[i] <= camarilla_pivot_aligned[i]
        
        # Signal logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals