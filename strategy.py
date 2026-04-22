#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-day Trend Filter and Volume Confirmation.
Long when price is above Alligator teeth (green line) and 1-day EMA50 is rising with volume spike.
Short when price is below Alligator teeth and 1-day EMA50 is falling with volume spike.
Exit when price crosses Alligator teeth or 1-day EMA50 reverses.
Williams Alligator uses SMAs of median price (H+L)/2 with specific periods to filter noise.
Designed for low trade frequency by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 1-day trend.
"""

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
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2
    # Jaw (blue): 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (red): 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (green): 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1-day close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price above teeth (lips > teeth), 1-day EMA50 rising, volume spike
            if (lips[i] > teeth[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below teeth (lips < teeth), 1-day EMA50 falling, volume spike
            elif (lips[i] < teeth[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses teeth or 1-day EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= teeth or 1-day EMA50 turns down
                if lips[i] <= teeth[i] or ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price >= teeth or 1-day EMA50 turns up
                if lips[i] >= teeth[i] or ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0