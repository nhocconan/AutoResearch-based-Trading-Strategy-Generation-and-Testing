#!/usr/bin/env python3
"""
1H_CAMARILLA_PIVOT_BREAKOUT_4HTREND_1DVOLUME
Hypothesis: Camarilla pivot points on 4h timeframe provide strong support/resistance levels.
Breakouts above R3 or below S3 with 1d volume confirmation and 4h trend filter yield high-probability
trades in both bull and bear markets. Uses 1h for entry timing only to keep trade frequency low.
Target: 20-40 trades/year (80-160 total over 4 years) on 1h timeframe.
"""
name = "1H_CAMARILLA_PIVOT_BREAKOUT_4HTREND_1DVOLUME"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + (range_val * 1.5000)
    r3 = c + (range_val * 1.2500)
    r2 = c + (range_val * 1.1666)
    r1 = c + (range_val * 1.0833)
    s1 = c - (range_val * 1.0833)
    s2 = c - (range_val * 1.1666)
    s3 = c - (range_val * 1.2500)
    s4 = c - (range_val * 1.5000)
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for Camarilla pivots and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each 4h bar
    r4_4h = np.zeros(len(df_4h))
    r3_4h = np.zeros(len(df_4h))
    s3_4h = np.zeros(len(df_4h))
    s4_4h = np.zeros(len(df_4h))
    
    for i in range(len(df_4h)):
        r4, r3, _, _, _, _, s3, s4 = calculate_camarilla(high_4h[i], low_4h[i], close_4h[i])
        r4_4h[i] = r4
        r3_4h[i] = r3
        s3_4h[i] = s3
        s4_4h[i] = s4
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h and 1d data to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike: current 1d volume > 1.5x 20-day average
        volume_spike = volume_1d_aligned[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above R3 with volume spike and uptrend
            if (high[i] > r3_4h_aligned[i] and 
                volume_spike and 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S3 with volume spike and downtrend
            elif (low[i] < s3_4h_aligned[i] and 
                  volume_spike and 
                  close[i] < ema50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below S3 or trend reversal
            if (low[i] < s3_4h_aligned[i] or 
                close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Break above R3 or trend reversal
            if (high[i] > r3_4h_aligned[i] or 
                close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals