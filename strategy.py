#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_v1
Hypothesis: 12-hour strategy using 1-day Camarilla pivot levels for entries, with volume confirmation and trend filter via 200-period EMA.
Works in bull/bear by requiring alignment with the 1d trend (EMA200) and confirming with volume to avoid false breakouts.
Targets 12-37 trades per year (50-150 total over 4 years) to minimize fee drag.
"""

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend direction
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Previous 1d bar's range for Camarilla
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    range_1d = prev_high_1d - prev_low_1d
    # Resistance levels
    r3 = prev_close_1d + range_1d * 1.1 / 2
    r4 = prev_close_1d + range_1d * 1.1
    # Support levels
    s3 = prev_close_1d - range_1d * 1.1 / 2
    s4 = prev_close_1d - range_1d * 1.1
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > EMA200 (uptrend) AND close breaks above R4 with volume
        if (close[i] > ema200_1d_aligned[i] and close[i] > r4_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < EMA200 (downtrend) AND close breaks below S4 with volume
        elif (close[i] < ema200_1d_aligned[i] and close[i] < s4_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals