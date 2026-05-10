#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot breakouts (R3/S3) with volume confirmation and 1d EMA trend filter produce profitable trades in both bull and bear markets by capturing momentum after rejection of key daily support/resistance levels.
Timeframe: 6h balances trade frequency and signal quality.
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current day (using previous day's OHLC)
    # Standard Camarilla formula
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First value uses current day's high as placeholder
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    R4 = prev_close + (prev_high - prev_low) * 1.1
    S4 = prev_close - (prev_high - prev_low) * 1.1
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get 1d data for EMA trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # 6h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 (34 periods)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA34
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current 6h volume > 1.5x 20-period average
        vol_ma20 = pd.Series(volume_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
        vol_ma20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma20)
        volume_filter = volume[i] > vol_ma20_aligned[i] * 1.5
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume and uptrend
            if high[i] > R3_aligned[i] and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with volume and downtrend
            elif low[i] < S3_aligned[i] and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R4 or trend fails
            if high[i] >= R4_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S4 or trend fails
            if low[i] <= S4_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals