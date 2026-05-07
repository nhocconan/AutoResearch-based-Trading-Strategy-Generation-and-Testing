#!/usr/bin/env python3
"""
12H_1D_Camarilla_Pivot_Bounce_With_Volume_Confirmation
Hypothesis: Price tends to respect daily Camarilla pivot levels (H3/L3) on 12h timeframe.
In bull markets: price above daily EMA50, look for long bounces from daily L3 with volume confirmation.
In bear markets: price below daily EMA50, look for short bounces from daily H3 with volume confirmation.
Uses strict entry conditions to limit trades and reduce fee drag, suitable for 12h timeframe.
"""
name = "12H_1D_Camarilla_Pivot_Bounce_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1D data for Camarilla levels and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily Camarilla levels (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H3 = close + (high - low) * 1.1/4, L3 = close - (high - low) * 1.1/4
    h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume filter: current 12h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily EMA50, touches or crosses above daily L3, volume confirmation
            if (close[i] > ema_50_1d_aligned[i] and 
                close[i] > l3_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA50, touches or crosses below daily H3, volume confirmation
            elif (close[i] < ema_50_1d_aligned[i] and 
                  close[i] < h3_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily EMA50
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily EMA50
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals