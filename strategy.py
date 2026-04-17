#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Camarilla pivot breakout + volume confirmation + 6h EMA50 trend filter.
Long when price breaks above 1d R3 level with volume > 1.4x 20-period average and price > 6h EMA50.
Short when price breaks below 1d S3 level with volume > 1.4x 20-period average and price < 6h EMA50.
Exit on opposite pivot level (S1 for long, R1 for short) or EMA cross reversal.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Volume confirmation filters low-activity breakouts. EMA50 ensures alignment with intermediate trend.
Works in bull markets (breakout continuation) and bear markets (mean reversion at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla levels (based on previous day)
    # Camarilla: Range = high - low
    # R4 = close + Range * 1.1/2
    # R3 = close + Range * 1.1/4
    # R2 = close + Range * 1.1/6
    # R1 = close + Range * 1.1/12
    # S1 = close - Range * 1.1/12
    # S2 = close - Range * 1.1/6
    # S3 = close - Range * 1.1/4
    # S4 = close - Range * 1.1/2
    range_1d = high_1d - low_1d
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # Get 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or
            np.isnan(ema50[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.4x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.4 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily R3 with volume and above 6h EMA50
            if (close[i] > r3_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S3 with volume and below 6h EMA50
            elif (close[i] < s3_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily S1 or EMA50 cross down
            if close[i] < s1_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily R1 or EMA50 cross up
            if close[i] > r1_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dCamarilla_R3S3_Volume_EMA50"
timeframe = "6h"
leverage = 1.0