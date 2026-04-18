#!/usr/bin/env python3
"""
12h Volume Spike + Daily Close Breakout with Trend Filter
Strategy: Enter long when price breaks above previous day's high with volume spike,
          short when price breaks below previous day's low with volume spike.
          Use daily EMA50 as trend filter to avoid counter-trend trades.
          Designed for low trade frequency with clear breakout edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for breakout levels and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily high, low, close for breakout levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily levels to 12h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or 
            np.isnan(daily_close_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        daily_high_level = daily_high_aligned[i]
        daily_low_level = daily_low_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: break above daily high with volume spike and above daily EMA50
            if (price > daily_high_level and volume_spike[i] and price > ema_50):
                signals[i] = 0.25
                position = 1
            # Short: break below daily low with volume spike and below daily EMA50
            elif (price < daily_low_level and volume_spike[i] and price < ema_50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below daily low or below daily EMA50
            if price < daily_low_level or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above daily high or above daily EMA50
            if price > daily_high_level or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_VolumeSpike_DailyBreakout_EMA50"
timeframe = "12h"
leverage = 1.0