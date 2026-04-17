#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Elder Ray (Bull/Bear Power) + 1w EMA50 trend filter + volume confirmation.
Long when Bull Power > 0, Bear Power < 0, price > 1w EMA50, and volume > 1.5x 20-period average.
Short when Bull Power < 0, Bear Power > 0, price < 1w EMA50, and volume > 1.5x 20-period average.
Exit when Elder Power signals reverse or volume drops.
Uses 1w for trend (reduces noise) and 6h for entry timing and Elder Ray calculation.
Designed to capture institutional buying/selling pressure with trend and volume confirmation in both bull and bear markets.
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
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Align 1w EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and Elder Ray
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > EMA50, volume confirmed
            if (bull_power_1d_aligned[i] > 0 and 
                bear_power_1d_aligned[i] < 0 and 
                close[i] > ema50_1w_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, price < EMA50, volume confirmed
            elif (bull_power_1d_aligned[i] < 0 and 
                  bear_power_1d_aligned[i] > 0 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Elder Power signals reverse OR volume drops
            if (bull_power_1d_aligned[i] <= 0 or 
                bear_power_1d_aligned[i] >= 0 or 
                not volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Elder Power signals reverse OR volume drops
            if (bull_power_1d_aligned[i] >= 0 or 
                bear_power_1d_aligned[i] <= 0 or 
                not volume_confirmed):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dElderRay_Volume_1wEMA50_Trend"
timeframe = "6h"
leverage = 1.0