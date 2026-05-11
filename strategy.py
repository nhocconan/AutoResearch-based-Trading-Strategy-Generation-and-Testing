#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Use 1d Camarilla pivot points (R1/S1) for breakout entries with 1d trend filter and volume spike confirmation.
Designed to capture institutional breakout attempts in both bull and bear markets while avoiding false breakouts.
Limits trades by requiring alignment between 1d trend, volume confirmation, and price action at key pivot levels.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot points for the day"""
    # Pivot point
    pivot = (high + low + close) / 3
    # Range
    range_ = high - low
    
    # Camarilla levels
    r4 = close + range_ * 1.500
    r3 = close + range_ * 1.250
    r2 = close + range_ * 1.166
    r1 = close + range_ * 1.083
    s1 = close - range_ * 1.083
    s2 = close - range_ * 1.166
    s3 = close - range_ * 1.250
    s4 = close - range_ * 1.500
    
    return r1, s1, r2, s2, r3, s3, r4, s4

def calculate_ema(arr, period):
    """Calculate EMA with proper handling of NaN"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (R1, S1)
    r1_1d, s1_1d, _, _, _, _, _, _ = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = calculate_ema(close_1d, 34)
    
    # Calculate volume spike: current 1d volume > 2.0x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    
    # Align 1d indicators to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R1 AND Uptrend (close > EMA34) AND Volume spike
            if close[i] > r1_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 AND Downtrend (close < EMA34) AND Volume spike
            elif close[i] < s1_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < S1 OR trend turns down (close < EMA34)
            if close[i] < s1_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Close > R1 OR trend turns up (close > EMA34)
            if close[i] > r1_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals