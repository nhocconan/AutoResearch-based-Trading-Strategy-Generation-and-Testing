#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA50 trend filter and 12h volume spike (>2.0x 20-bar MA). Uses 12h HTF for stronger trend confirmation than 4h alone, reducing false signals. Volume spike confirms institutional interest. Discrete sizing 0.25 to limit fee drift. Target 25-50 trades/year on 4h timeframe. Works in bull/bear via trend filter + volume confirmation.
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
    
    # Get 12h data for HTF trend (EMA50) and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA50 on 12h for HTF trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-bar volume MA on 12h for volume spike detection
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (2.0 * vol_ma_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    # Calculate Camarilla levels from previous 4h bar (for 4h entry timing)
    camarilla_range = (high - low) * 1.1 / 12.0
    camarilla_R1 = close + camarilla_range
    camarilla_S1 = close - camarilla_range
    
    # Shift by 1 to use only completed 4h bar for Camarilla calculation (no look-ahead)
    camarilla_R1 = np.roll(camarilla_R1, 1)
    camarilla_S1 = np.roll(camarilla_S1, 1)
    camarilla_R1[0] = np.nan
    camarilla_S1[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (20), and Camarilla (1)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_spike_12h_aligned[i]) or 
            np.isnan(camarilla_R1[i]) or 
            np.isnan(camarilla_S1[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + above 12h EMA50 + 12h volume spike
            long_setup = (close[i] > camarilla_R1[i]) and \
                         (close[i] > ema_50_12h_aligned[i]) and \
                         volume_spike_12h_aligned[i]
            # Short: price breaks below Camarilla S1 + below 12h EMA50 + 12h volume spike
            short_setup = (close[i] < camarilla_S1[i]) and \
                          (close[i] < ema_50_12h_aligned[i]) and \
                          volume_spike_12h_aligned[i]
            
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
            if (close[i] < camarilla_S1[i]) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Camarilla R1 OR above 12h EMA50
            if (close[i] > camarilla_R1[i]) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0