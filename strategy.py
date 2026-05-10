#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, trade breakouts of Donchian(20) channels with 1d EMA50 trend filter and volume spike confirmation. Uses 12h timeframe to reduce trade frequency and capture multi-day moves, with volume confirmation to filter false breakouts. Target: 15-30 trades/year for better generalization to bear markets.
"""

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get daily data for Donchian channel (20-period high/low)
    df_1d_donch = get_htf_data(prices, '1d')
    if len(df_1d_donch) < 20:
        return np.zeros(n)
    
    high_1d = df_1d_donch['high'].values
    low_1d = df_1d_donch['low'].values
    
    # Calculate Donchian(20) from previous day (to avoid look-ahead)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d_donch, high_max_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d_donch, low_min_20)
    
    # Get 12h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track bars since last entry
    
    # Warmup: need 1d EMA (50) and Donchian (20 + 1 for shift)
    start_idx = 70  # 50 for EMA + 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                bars_since_entry = 0
            continue
        
        # Increment bars since entry if in a position
        if position != 0:
            bars_since_entry += 1
        
        # Trend filter: price vs 1d EMA50
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high in uptrend with volume spike
            if high[i] > donch_high_aligned[i] and uptrend_1d and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below Donchian low in downtrend with volume spike
            elif low[i] < donch_low_aligned[i] and downtrend_1d and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Long exit: price drops below Donchian low or trend fails
            if low[i] < donch_low_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Donchian high or trend fails
            if high[i] > donch_high_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals