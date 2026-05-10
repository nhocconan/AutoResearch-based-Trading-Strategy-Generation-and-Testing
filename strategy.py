#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Camarilla pivot levels from 12h timeframe provide high-probability support/resistance levels.
# Breakouts above R3 or below S3 with volume confirmation and 12h trend filter (EMA34) capture strong momentum moves.
# Designed for 4h timeframe to balance trade frequency and signal quality, targeting 20-40 trades/year.
# Works in both bull and bear markets by following the 12h trend direction and requiring volume confirmation.

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    multiplier = 1.1 / 12
    R4 = close + range_val * multiplier * 11
    R3 = close + range_val * multiplier * 6
    R2 = close + range_val * multiplier * 4
    R1 = close + range_val * multiplier * 2
    PP = (high + low + close) / 3
    S1 = close - range_val * multiplier * 2
    S2 = close - range_val * multiplier * 4
    S3 = close - range_val * multiplier * 6
    S4 = close - range_val * multiplier * 11
    return S3, S2, S1, PP, R1, R2, R3, R4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get 12h data for trend and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Camarilla levels from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    s3_12h = np.zeros_like(close_12h)
    r3_12h = np.zeros_like(close_12h)
    
    for i in range(len(close_12h)):
        s3, s2, s1, pp, r1, r2, r3, r4 = camarilla(high_12h[i], low_12h[i], close_12h[i])
        s3_12h[i] = s3
        r3_12h[i] = r3
    
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    
    # Get 4h data for entry and volume
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) + volume EMA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or
            np.isnan(r3_12h_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above R3 + uptrend (price > EMA34) + volume
            if close[i] > r3_12h_aligned[i] and close[i] > ema34_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below S3 + downtrend (price < EMA34) + volume
            elif close[i] < s3_12h_aligned[i] and close[i] < ema34_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below EMA34 or volume drops
            if close[i] < ema34_12h_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above EMA34 or volume drops
            if close[i] > ema34_12h_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals