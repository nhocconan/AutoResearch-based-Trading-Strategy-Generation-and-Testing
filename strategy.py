#!/usr/bin/env python3
"""
12h_1d_donchian_breakout_volume_filter
Hypothesis: 12-hour Donchian breakout strategy with daily trend filter and volume confirmation.
Uses daily Donchian channels (20-period) for breakout direction, confirmed by daily EMA50 trend and volume spike.
Designed to capture strong trending moves while avoiding chop via volume filter. Target: 15-25 trades/year (60-100 total) to minimize fee drag.
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
    
    # Get daily data for trend and breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average (20-period) for volume spike detection
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: require volume above 1.5x average to avoid low-vol breakouts
        volume_spike = volume[i] > (vol_avg_20_aligned[i] * 1.5)
        
        # Breakout conditions with trend filter
        bullish_breakout = (close[i] > donch_high_20_aligned[i]) and volume_spike
        bearish_breakout = (close[i] < donch_low_20_aligned[i]) and volume_spike
        
        # Trend filter: only take breakouts in direction of daily EMA50 trend
        if bullish_breakout and ema50_1d_aligned[i] > close[i]:  # Uptrend: price above EMA50
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif bearish_breakout and ema50_1d_aligned[i] < close[i]:  # Downtrend: price below EMA50
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Hold position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_volume_filter"
timeframe = "12h"
leverage = 1.0