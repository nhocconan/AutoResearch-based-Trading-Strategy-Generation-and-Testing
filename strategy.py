#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla R3/S3 levels from 1d provide strong intraday support/resistance.
Breakouts above R3 or below S3 with volume confirmation and 1d EMA trend filter capture
continuation moves in both bull and bear markets. Designed for 15-30 trades/year on 6h.
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla equations
    range_prev = high_prev - low_prev
    camarilla_r3 = close_prev + range_prev * 1.1 / 4
    camarilla_s3 = close_prev - range_prev * 1.1 / 4
    camarilla_r4 = close_prev + range_prev * 1.1 / 2
    camarilla_s4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for daily bar to close)
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla (needs 1 day), EMA34 (34 bars), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or
            np.isnan(r4_6h[i]) or
            np.isnan(s4_6h[i]) or
            np.isnan(ema_34_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above R3 with volume
            if close[i] > ema_34_6h[i] and high[i] > r3_6h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below S3 with volume
            elif close[i] < ema_34_6h[i] and low[i] < s3_6h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below EMA34 OR reaches R4 (take profit)
            if close[i] < ema_34_6h[i] or high[i] >= r4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above EMA34 OR reaches S4 (take profit)
            if close[i] > ema_34_6h[i] or low[i] <= s4_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals