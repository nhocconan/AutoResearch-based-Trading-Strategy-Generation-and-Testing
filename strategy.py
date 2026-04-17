#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Elder Ray (Bull/Bear Power) + volume confirmation + 6h EMA50 trend filter.
Long when Bull Power > 0, volume > 1.5x 20-period average, and close > 6h EMA50 (uptrend).
Short when Bear Power < 0, volume > 1.5x 20-period average, and close < 6h EMA50 (downtrend).
Exit when Elder Power crosses zero (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit).
Elder Ray measures buying/selling pressure relative to EMA13, providing early trend strength signals.
Designed to work in both bull and bear markets by capturing institutional volume-driven moves while avoiding false signals in low-volume chop.
Uses 1d timeframe for Elder Ray calculation (reduces noise) and 6h for entry timing and trend confirmation.
Target trades: 12-37 per year over 4 years (50-150 total).
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
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA13
    
    # Calculate 6h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Elder Ray components to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure), volume confirmed, and uptrend (close > EMA50)
            if (bull_power_1d_aligned[i] > 0 and 
                volume_confirmed and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure), volume confirmed, and downtrend (close < EMA50)
            elif (bear_power_1d_aligned[i] < 0 and 
                  volume_confirmed and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 (buying pressure faded)
            if bull_power_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 (selling pressure faded)
            if bear_power_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dElderRay_Volume_EMA50_Trend"
timeframe = "6h"
leverage = 1.0