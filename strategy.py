#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla R3/S3 breakout on 4h with daily trend filter (price > EMA50 for longs, < EMA50 for shorts) and volume spike confirmation. Works in bull markets by catching breakouts and in bear markets by fading false breaks via trend filter. Volume confirms institutional participation. Designed for 4h timeframe to target 25-50 trades/year.
"""

name = "4h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 2
    s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike and above daily EMA50
            if close[i] > r3_aligned[i] and vol_spike and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and below daily EMA50
            elif close[i] < s3_aligned[i] and vol_spike and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or volume drops
            if close[i] < s3_aligned[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or volume drops
            if close[i] > r3_aligned[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals