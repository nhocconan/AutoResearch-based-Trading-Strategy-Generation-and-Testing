#!/usr/bin/env python3
name = "6H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 24:  # Need at least 24 hours (4 bars) of 6h data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3, S3, R4, S4 levels
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # R4 = close + (high - low) * 1.1
    # S4 = close - (high - low) * 1.1
    range_hl = prev_high - prev_low
    r3 = prev_close + range_hl * 1.1 / 2
    s3 = prev_close - range_hl * 1.1 / 2
    r4 = prev_close + range_hl * 1.1
    s4 = prev_close - range_hl * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: current volume > 2x 24-period average (4 days)
    avg_volume = np.convolve(volume, np.ones(24)/24, mode='same')
    # Handle edges properly
    for i in range(24):
        if i < 12:
            avg_volume[i] = np.mean(volume[:i+12]) if i+12 < len(volume) else np.mean(volume)
        elif i >= len(volume) - 12:
            avg_volume[i] = np.mean(volume[i-12:]) if i-12 >= 0 else np.mean(volume)
    
    volume_spike = volume > avg_volume * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with uptrend and volume spike
            if close[i] > r3_aligned[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with downtrend and volume spike
            elif close[i] < s3_aligned[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (reversion to mean)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 (reversion to mean)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals