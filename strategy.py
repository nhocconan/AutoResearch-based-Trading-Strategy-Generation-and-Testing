#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_v1
# Hypothesis: Uses Camarilla pivot levels from 1-day timeframe with volume confirmation.
# Long when price breaks above R3 with volume > 1.5x average; short when breaks below S3.
# Designed to work in both bull and bear markets by capturing breakouts from key intraday levels.
# Target: 15-25 trades/year (60-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate pivot points and levels for each 1-day bar
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 12h timeframe (wait for 1-day bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price closes below pivot (mean reversion) or stop loss
            if close[i] < pivot[i-20 if i>=20 else 0]:  # Use previous day's pivot
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above pivot (mean reversion) or stop loss
            if close[i] > pivot[i-20 if i>=20 else 0]:  # Use previous day's pivot
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R3 with volume confirmation
            if close[i] > r3_aligned[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S3 with volume confirmation
            elif close[i] < s3_aligned[i] and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals