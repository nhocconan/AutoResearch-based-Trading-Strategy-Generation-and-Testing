#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: On 12h timeframe, trade breakouts of 20-period Donchian channels with 1d EMA trend filter and volume spike confirmation. This captures institutional breakouts with trend alignment, reducing false signals. Works in bull/bear via trend filter and volume confirmation to avoid chop. Target: 15-30 trades/year.
"""

name = "12h_Donchian20_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    
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
    
    # Warmup: need 1d EMA (50) and Donchian (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(high_max_20_aligned[i]) or
            np.isnan(low_min_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA50
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high in uptrend with volume spike
            if high[i] > high_max_20_aligned[i] and uptrend_1d and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in downtrend with volume spike
            elif low[i] < low_min_20_aligned[i] and downtrend_1d and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below Donchian low or trend fails
            if low[i] < low_min_20_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Donchian high or trend fails
            if high[i] > high_max_20_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals