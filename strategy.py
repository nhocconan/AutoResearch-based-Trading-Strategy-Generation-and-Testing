#!/usr/bin/env python3
"""
#100995 - 6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels derived from 1d candles provide significant intraday support/resistance. 
Breakout beyond R3/S3 with 1-week trend confirmation (price above/below weekly EMA34) and volume surge 
indicates strong momentum continuation. Works in bull markets (breakouts with trend) and bear markets 
(breakdowns with trend). Uses 6h timeframe to balance trade frequency and signal quality.
"""

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d candle
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # R2 = close + 0.5 * (high - low)
    # R1 = close + 0.25 * (high - low)
    # S1 = close - 0.25 * (high - low)
    # S2 = close - 0.5 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_calc = lambda h, l, c: (
        c + 1.5 * (h - l),  # R4
        c + 1.0 * (h - l),  # R3
        c + 0.5 * (h - l),  # R2
        c + 0.25 * (h - l), # R1
        c - 0.25 * (h - l), # S1
        c - 0.5 * (h - l),  # S2
        c - 1.0 * (h - l),  # S3
        c - 1.5 * (h - l)   # S4
    )
    
    # Calculate for each 1d bar
    camarilla_levels = [camarilla_calc(h, l, c) for h, l, c in zip(high_1d, low_1d, close_1d)]
    r4_1d, r3_1d, r2_1d, r1_1d, s1_1d, s2_1d, s3_1d, s4_1d = zip(*camarilla_levels) if camarilla_levels else ([],)*8
    
    r4_1d = np.array(r4_1d)
    r3_1d = np.array(r3_1d)
    r2_1d = np.array(r2_1d)
    r1_1d = np.array(r1_1d)
    s1_1d = np.array(s1_1d)
    s2_1d = np.array(s2_1d)
    s3_1d = np.array(s3_1d)
    s4_1d = np.array(s4_1d)
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1w data for trend filter (weekly EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 2.0x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R3, above weekly EMA34, volume surge
        if (close[i] > r3_1d_aligned[i] and 
            close[i] > ema34_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S3, below weekly EMA34, volume surge
        elif (close[i] < s3_1d_aligned[i] and 
              close[i] < ema34_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite S1/R1 level (mean reversion)
        elif position == 1 and close[i] < s1_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0