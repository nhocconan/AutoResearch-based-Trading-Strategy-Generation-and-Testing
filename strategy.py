#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 6h timeframe using Camarilla pivot levels (R3, S3) from prior day for entry, 
1d EMA50 for trend filter, and 6h volume spike (>2.0x 20-bar MA) for confirmation. 
Enter long when price breaks above Camarilla R3 AND above 1d EMA50 AND volume spike. 
Enter short when price breaks below Camarilla S3 AND below 1d EMA50 AND volume spike. 
Exit on opposite Camarilla touch (S3 for long, R3 for short) or trend reversal. 
Uses discrete sizing 0.25 to balance return and drawdown. Target 12-30 trades/year on 6h timeframe. 
Camarilla R3/S3 levels represent stronger breakout points than R1/S1, reducing false signals. 
The 1d EMA50 filter ensures we only trade with the daily trend, improving performance in both bull and bear markets. 
Volume confirmation avoids breakouts from low participation. 
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior day: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3_1d = close_1d + (1.1 * (high_1d - low_1d) / 4)
    camarilla_s3_1d = close_1d - (1.1 * (high_1d - low_1d) / 4)
    
    # Align Camarilla levels to 6h timeframe (prior day's levels available at 00:00 UTC)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Get 1d data for EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-bar volume MA on 6h for volume spike detection
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (2.0 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA50 (50) and 6h volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND above 1d EMA50 AND volume spike
            long_setup = (close[i] > camarilla_r3_1d_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_spike_6h[i]
            # Short: price breaks below Camarilla S3 AND below 1d EMA50 AND volume spike
            short_setup = (close[i] < camarilla_s3_1d_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_spike_6h[i]
            
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
            # Exit: price touches Camarilla S3 OR closes below 1d EMA50
            if (close[i] <= camarilla_s3_1d_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R3 OR closes above 1d EMA50
            if (close[i] >= camarilla_r3_1d_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0