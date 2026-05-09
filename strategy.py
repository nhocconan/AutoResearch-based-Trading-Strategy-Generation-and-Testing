#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Donchian channels provide clear breakout signals, 1d EMA50 filters trend direction,
# volume surge confirms institutional participation. Designed to work in both bull and bear
# markets by capturing strong momentum moves. Target: 25-40 trades/year (100-160 over 4 years)
# to minimize fee drag while maintaining sufficient statistical significance.
name = "12h_Donchian_20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 12h data
    # We need to calculate this manually since we don't have 12h data directly
    # Instead, we'll use price action and wait for 12h bar closes via alignment
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 12h
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_1d_12h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Price breaks above Donchian upper + above 1d EMA50 + volume spike
            if close[i] > high_max[i] and close[i] > ema50_1d_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + below 1d EMA50 + volume spike
            elif close[i] < low_min[i] and close[i] < ema50_1d_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Donchian lower OR below 1d EMA50
            if close[i] < low_min[i] or close[i] < ema50_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Donchian upper OR above 1d EMA50
            if close[i] > high_max[i] or close[i] > ema50_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals