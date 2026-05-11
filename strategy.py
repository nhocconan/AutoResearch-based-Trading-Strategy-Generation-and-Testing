#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Use daily trend (EMA34) as filter, Camarilla R3/S3 levels for breakout entries,
and volume spike for confirmation. Works in both bull and bear markets by trading breakouts
in the direction of the daily trend. Target: 50-150 total trades over 4 years on 6h timeframe.
"""

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D Data for Trend and Camarilla Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Previous day's Camarilla levels (using prior day's OHLC)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # First day uses current day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla calculation
    range_prev = prev_high - prev_low
    camarilla_mult = 1.1 / 6
    R3 = prev_close + range_prev * camarilla_mult * 3
    S3 = prev_close - range_prev * camarilla_mult * 3
    R4 = prev_close + range_prev * camarilla_mult * 4
    S4 = prev_close - range_prev * camarilla_mult * 4
    
    # Align 1D data to 6H
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine daily trend: above EMA34 = uptrend, below = downtrend
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R3/R4 with volume spike in uptrend
            if uptrend and vol_spike[i] and (close[i] > R3_aligned[i] or close[i] > R4_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3/S4 with volume spike in downtrend
            elif downtrend and vol_spike[i] and (close[i] < S3_aligned[i] or close[i] < S4_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 or trend changes
            if close[i] < S3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 or trend changes
            if close[i] > R3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals