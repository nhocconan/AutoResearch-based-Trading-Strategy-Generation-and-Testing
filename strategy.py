#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA100 trend and volume spike
# Uses daily EMA100 for trend filter, 12h Donchian channel for breakout signals, and volume spike for confirmation.
# Works in both bull and bear markets by requiring trend alignment and volume confirmation.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
name = "12h_Donchian20_1dEMA100_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA100 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # 1d EMA100 for trend filter
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_12h = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Get 12h data for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Donchian(20) channel
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper and lower bands
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h (they're already in 12h, but we need to align for consistency)
    # Since we're using 12h data directly, no alignment needed for Donchian
    # But we'll use the values as-is from df_12h
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # Ensure enough data for EMA100 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema100_12h[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above Donchian upper with uptrend and volume spike
            if close[i] > donchian_upper[i] and close[i] > ema100_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower with downtrend and volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema100_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Donchian lower OR trend turns down
            if close[i] < donchian_lower[i] or close[i] < ema100_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Donchian upper OR trend turns up
            if close[i] > donchian_upper[i] or close[i] > ema100_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals