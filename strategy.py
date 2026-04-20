#!/usr/bin/env python3
"""
6h_12h_Camarilla_R3S3_Breakout_Volume
Hypothesis: Camarilla pivot levels (R3/S3) from 12h act as strong support/resistance in ranging markets (2025+).
Breakouts above R3 or below S3 with volume confirmation indicate institutional interest and trend continuation.
Uses 12h for pivot levels and trend filter, 6s for entry timing and volume confirmation.
Works in bull markets (breakouts continue up) and bear markets (breakdowns continue down).
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25 to manage drawdown.
"""

name = "6h_12h_Camarilla_R3S3_Breakout_Volume"
timeframe = "6h"
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    def calculate_camarilla(high, low, close):
        # Typical price
        tp = (high + low + close) / 3
        # Range
        r = high - low
        
        # Camarilla levels
        r3 = close + (r * 1.1000 / 4)
        s3 = close - (r * 1.1000 / 4)
        r4 = close + (r * 1.1000 / 2)
        s4 = close - (r * 1.1000 / 2)
        
        return r3, s3, r4, s4
    
    r3_12h, s3_12h, r4_12h, s4_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate volume average (20-period) for volume spike detection
    vol_avg = np.zeros_like(volume)
    vol_avg[:] = np.nan
    for i in range(20, len(volume)):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_avg[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_spike = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long breakout: price closes above R3 with volume spike
            if close[i] > r3_12h_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S3 with volume spike
            elif close[i] < s3_12h_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below R3 (failed breakout) or reverses below S3
            if close[i] < r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above S3 (failed breakdown) or reverses above R3
            if close[i] > s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals