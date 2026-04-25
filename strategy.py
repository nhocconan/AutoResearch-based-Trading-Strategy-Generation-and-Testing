#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 12h timeframe using Camarilla pivot levels (R1/S1) from 1d for entry, 
daily EMA50 for trend filter, and daily volume spike (>2.0x 20-bar MA) for confirmation. 
Enter long when price > 1d Camarilla R1 AND above daily EMA50 AND volume spike. 
Enter short when price < 1d Camarilla S1 AND below daily EMA50 AND volume spike. 
Exit on opposite pivot touch (S1 for long, R1 for short) or trend reversal (close crosses EMA50). 
Uses discrete sizing 0.25 to balance return and drawdown. Target 12-37 trades/year on 12h timeframe. 
Works in bull/bear via 1d trend filter and volume confirmation to avoid false breakouts.
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
    
    # Get 1d data for Camarilla pivot points (R1, S1), EMA50 trend filter, and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot points: 
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + (camarilla_range * 1.1 / 12)
    s1_1d = close_1d - (camarilla_range * 1.1 / 12)
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d bar only)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d Camarilla R1 AND above daily EMA50 AND volume spike
            long_setup = (close[i] > r1_1d_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price below 1d Camarilla S1 AND below daily EMA50 AND volume spike
            short_setup = (close[i] < s1_1d_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i]
            
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
            # Exit: price touches 1d Camarilla S1 OR closes below daily EMA50
            if (close[i] <= s1_1d_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 1d Camarilla R1 OR closes above daily EMA50
            if (close[i] >= r1_1d_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0