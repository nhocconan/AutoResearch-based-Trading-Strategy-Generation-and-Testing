#!/usr/bin/env python3
name = "12h_1W_Camarilla_R4_S4_Breakout_1W_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Camarilla levels
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    # Range = H - L
    range_1w = high_1w - low_1w
    # Resistance levels
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    r2_1w = close_1w + (range_1w * 1.1 / 6)
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    # Support levels
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    s2_1w = close_1w - (range_1w * 1.1 / 6)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    
    # Weekly EMA8 for trend filter
    ema8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Align weekly data to 12h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    ema8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema8_1w)
    
    # 12h volume spike detection (volume > 1.5 * 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(ema8_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R4 with volume spike and weekly uptrend
            if (close[i] > r4_1w_aligned[i] and 
                volume_spike[i] and
                ema8_1w_aligned[i] > ema8_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 with volume spike and weekly downtrend
            elif (close[i] < s4_1w_aligned[i] and 
                  volume_spike[i] and
                  ema8_1w_aligned[i] < ema8_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below R3 or weekly trend turns down
            if (close[i] < r3_1w_aligned[i] or 
                ema8_1w_aligned[i] < ema8_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above S3 or weekly trend turns up
            if (close[i] > s3_1w_aligned[i] or 
                ema8_1w_aligned[i] > ema8_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals