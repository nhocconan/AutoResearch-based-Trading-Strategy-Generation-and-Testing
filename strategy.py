#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_Pivot_R1S1_Breakout_Volume
Hypothesis: Trade breakouts above/below 4h and 1d Camarilla R1/S1 levels with volume confirmation. 
Use 4h/1d for direction (trend filter) and 1h for precise entry timing. 
Position size 0.20 to limit risk and reduce trade frequency. 
Session filter (08-20 UTC) to avoid low-liquidity hours. 
Designed to work in bull/bear by capturing breakouts with institutional volume support.
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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 4h calculations (previous bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar's OHLC (completed bar)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    prev_close_4h[0] = close_4h[0]
    
    # 4h Camarilla levels (based on previous bar)
    R1_4h = np.full_like(high_4h, np.nan)
    S1_4h = np.full_like(low_4h, np.nan)
    
    for i in range(1, len(high_4h)):
        range_ = prev_high_4h[i] - prev_low_4h[i]
        R1_4h[i] = prev_close_4h[i] + range_ * 1.1 / 12
        S1_4h[i] = prev_close_4h[i] - range_ * 1.1 / 12
    
    # 1d calculations (previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's OHLC (completed day)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # 1d Camarilla levels (based on previous day)
    R1_1d = np.full_like(high_1d, np.nan)
    S1_1d = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        range_ = prev_high_1d[i] - prev_low_1d[i]
        R1_1d[i] = prev_close_1d[i] + range_ * 1.1 / 12
        S1_1d[i] = prev_close_1d[i] - range_ * 1.1 / 12
    
    # Align 4h Camarilla levels to 1h timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # Align 1d Camarilla levels to 1h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above both 4h R1 and 1d R1 with volume
            if close[i] > R1_4h_aligned[i] and close[i] > R1_1d_aligned[i] and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below both 4h S1 and 1d S1 with volume
            elif close[i] < S1_4h_aligned[i] and close[i] < S1_1d_aligned[i] and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price closes below either 4h S1 or 1d S1
            if close[i] < S1_4h_aligned[i] or close[i] < S1_1d_aligned[i]:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price closes above either 4h R1 or 1d R1
            if close[i] > R1_4h_aligned[i] or close[i] > R1_1d_aligned[i]:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0