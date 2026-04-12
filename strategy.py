#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_v1
Hypothesis: 4H breakout above/below daily Camarilla R4/S4 levels with volume confirmation.
Long when price breaks above R4 and volume > 1.5x 20-period average volume.
Short when price breaks below S4 and volume > 1.5x 20-period average volume.
Exit when price returns to daily midpoint (R4+S4)/2.
Uses daily timeframe for structure, 4H for entries to reduce whipsaw and overtrading.
Designed for 4H timeframe with low trade frequency (target: 20-50 trades/year).
Works in both bull and bear markets by following breakouts with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r4 = prev_close + range_ * 1.1 / 2
    camarilla_s4 = prev_close - range_ * 1.1 / 2
    camarilla_mid = (camarilla_r4 + camarilla_s4) / 2
    
    # Handle invalid ranges
    valid_range = range_ > 0
    camarilla_r4 = np.where(valid_range, camarilla_r4, np.nan)
    camarilla_s4 = np.where(valid_range, camarilla_s4, np.nan)
    camarilla_mid = np.where(valid_range, camarilla_mid, np.nan)
    
    # Align to 4H timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Volume confirmation: 20-period average volume on 4H chart
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(volume_threshold[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume confirmation
        long_breakout = high[i] > camarilla_r4_aligned[i] and volume[i] > volume_threshold[i]
        short_breakout = low[i] < camarilla_s4_aligned[i] and volume[i] > volume_threshold[i]
        
        # Exit conditions: return to Camarilla midpoint
        long_exit = close[i] < camarilla_mid_aligned[i]
        short_exit = close[i] > camarilla_mid_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals