#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) from daily timeframe provide significant support/resistance.
# Breakout above R3 or below S3 with volume confirmation and daily EMA34 trend filter.
# Works in bull markets via breakouts above resistance and in bear via breakdowns below support.
# Low trade frequency expected due to multiple confirmation layers.
# Target: 15-35 trades/year on 12h timeframe.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    close_val = close
    R4 = close_val + range_val * 1.500
    R3 = close_val + range_val * 1.250
    R2 = close_val + range_val * 1.166
    R1 = close_val + range_val * 1.083
    PP = (high + low + close_val) / 3
    S1 = close_val - range_val * 1.083
    S2 = close_val - range_val * 1.166
    S3 = close_val - range_val * 1.250
    S4 = close_val - range_val * 1.500
    return R3, R3, R2, R1, PP, S1, S2, S3  # Return R3 and S3 for breakout

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d = np.zeros(len(close_1d))
    s3_1d = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i < 1:  # Need at least one period for calculation
            r3_1d[i] = close_1d[i]
            s3_1d[i] = close_1d[i]
        else:
            r3, _, _, _, _, _, _, s3 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
            r3_1d[i] = r3
            s3_1d[i] = s3
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 12h timeframe (no look-ahead)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 12h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA34 (34) + volume EMA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above daily EMA34 AND volume confirmation
            if close[i] > r3_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below daily EMA34 AND volume confirmation
            elif close[i] < s3_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 or below daily EMA34
            if close[i] < s3_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or above daily EMA34
            if close[i] > r3_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals