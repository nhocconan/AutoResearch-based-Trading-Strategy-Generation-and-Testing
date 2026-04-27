#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: Donchian(20) breakout on 6h with weekly pivot direction filter (from 1w data) and volume confirmation.
Weekly pivot provides long-term bias (bull/bear) to avoid counter-trend trades. Volume spike filters false breakouts.
Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year) with low frequency to minimize fee drag.
Works in both bull and bear markets by aligning with weekly trend direction.
"""

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
    
    # Get weekly data for pivot direction (long-term bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard pivot: P = (H + L + C)/3
    # Support 1: S1 = (2*P) - H
    # Resistance 1: R1 = (2*P) - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pw = (high_1w + low_1w + close_1w) / 3.0
    r1w = (2 * pw) - high_1w
    s1w = (2 * pw) - low_1w
    
    # Weekly trend: price above R1 = bullish, below S1 = bearish, between = neutral
    bullish_week = close_1w > r1w
    bearish_week = close_1w < s1w
    
    # Align weekly bias to 6h timeframe (wait for weekly close)
    bullish_week_aligned = align_htf_to_ltf(prices, df_1w, bullish_week.astype(float))
    bearish_week_aligned = align_htf_to_ltf(prices, df_1w, bearish_week.astype(float))
    
    # Get daily data for Donchian calculation (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on daily timeframe (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-day high and low
    dh_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_1d, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1d, dl_20)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or \
           np.isnan(bullish_week_aligned[i]) or np.isnan(bearish_week_aligned[i]):
            signals[i] = 0.0
            continue
        
        donchian_high = dh_20_aligned[i]
        donchian_low = dl_20_aligned[i]
        is_bullish_week = bullish_week_aligned[i] > 0.5
        is_bearish_week = bearish_week_aligned[i] > 0.5
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: break above Donchian high with weekly bullish bias and volume spike
            if close[i] > donchian_high and is_bullish_week and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: break below Donchian low with weekly bearish bias and volume spike
            elif close[i] < donchian_low and is_bearish_week and vol_spike_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low or weekly bias turns bearish
            if close[i] < donchian_low or is_bearish_week:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above Donchian high or weekly bias turns bullish
            if close[i] > donchian_high or is_bullish_week:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0