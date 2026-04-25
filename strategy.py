#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume spike (>2.0x 20-bar MA). Uses 12h HTF for stronger trend confirmation than 1d, reducing false signals in sideways markets. Discrete sizing 0.25 to limit fee drag. Target 20-50 trades/year on 4h timeframe. Works in bull/bear via trend filter + volume confirmation.
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
    
    # Get 12h data for HTF trend and Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from previous 12h bar
    camarilla_range = (high_12h - low_12h) * 1.1 / 12.0
    camarilla_R1 = close_12h + camarilla_range
    camarilla_S1 = close_12h - camarilla_range
    
    # Align Camarilla levels to 4h (completed 12h bar only)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S1)
    
    # Calculate EMA50 on 12h for HTF trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h (completed 12h bar only)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Camarilla (12h), EMA50 (12h), volume MA (20)
    start_idx = max(20, 50)  # 50 for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + above 12h EMA50 + volume spike
            long_setup = (close[i] > camarilla_R1_aligned[i]) and \
                         (close[i] > ema_50_12h_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Camarilla S1 + below 12h EMA50 + volume spike
            short_setup = (close[i] < camarilla_S1_aligned[i]) and \
                          (close[i] < ema_50_12h_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below Camarilla S1 OR below 12h EMA50
            if (close[i] < camarilla_S1_aligned[i]) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Camarilla R1 OR above 12h EMA50
            if (close[i] > camarilla_R1_aligned[i]) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0