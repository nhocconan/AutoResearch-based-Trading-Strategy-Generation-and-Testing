#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_Volume_SessionFilter
1h strategy using 4h Camarilla pivot levels for direction and 1h for entry timing.
- Long: Price crosses above R1 level (4h) with volume > 1.5x 20-period average and in active session (08-20 UTC)
- Short: Price crosses below S1 level (4h) with volume > 1.5x 20-period average and in active session
- Exit: Opposite cross or session end
- Position size: 0.20 (20% of capital)
Designed for ~15-30 trades/year per symbol (60-120 total over 4 years)
Uses 4h for signal direction, 1h only for entry timing to reduce whipsaw.
Session filter reduces noise during low-volume hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels (direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h data
    r1_4h = np.zeros(len(df_4h))
    s1_4h = np.zeros(len(df_4h))
    for i in range(len(df_4h)):
        r1, s1 = calculate_camarilla(df_4h['high'].iloc[i], df_4h['low'].iloc[i], df_4h['close'].iloc[i])
        r1_4h[i] = r1
        s1_4h[i] = s1
    
    # Align 4h Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Price cross levels
        cross_above_r1 = close[i] > r1_4h_aligned[i] and close[i-1] <= r1_4h_aligned[i-1]
        cross_below_s1 = close[i] < s1_4h_aligned[i] and close[i-1] >= s1_4h_aligned[i-1]
        
        if position == 0:
            # Look for new entries only in session with volume confirmation
            if in_session and vol_confirm:
                if cross_above_r1:
                    signals[i] = 0.20
                    position = 1
                elif cross_below_s1:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: cross below S1 or session end
            if cross_below_s1 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: cross above R1 or session end
            if cross_above_r1 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0