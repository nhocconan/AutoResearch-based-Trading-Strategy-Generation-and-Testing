#!/usr/bin/env python3
# [24971] 6h_1d_pivot_fade_breakout_v1
# Hypothesis: 6-hour Camarilla pivot strategy combining fade at inner levels (R3/S3) and breakout at outer levels (R4/S4).
# Uses 1-day Camarilla pivots calculated from prior day's range.
# Fade logic: When price reaches R3 or S3 with overbought/oversold RSI(14) (<30 or >70), take opposite position.
# Breakout logic: When price breaks R4 or S4 with volume > 1.5x average, continue in breakout direction.
# Exit when price returns to the 1-day pivot point (central pivot).
# Designed for 6H timeframe to work in both trending and ranging markets across bull/bear cycles.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_pivot_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each day using prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize pivot levels arrays (same length as daily data)
    pivot = np.full(len(close_1d), np.nan)
    r4 = np.full(len(close_1d), np.nan)
    r3 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    s4 = np.full(len(close_1d), np.nan)
    
    # Calculate pivots starting from index 1 (need prior day's data)
    for i in range(1, len(close_1d)):
        # Use prior day's OHLC to calculate today's pivots
        ph = high_1d[i-1]  # prior day high
        pl = low_1d[i-1]   # prior day low
        pc = close_1d[i-1] # prior day close
        
        pivot[i] = (ph + pl + pc) / 3.0
        range_val = ph - pl
        # Camarilla formulas
        r4[i] = pc + range_val * 1.1 / 2.0
        r3[i] = pc + range_val * 1.1 / 4.0
        s3[i] = pc - range_val * 1.1 / 4.0
        s4[i] = pc - range_val * 1.1 / 2.0
    
    # Calculate RSI(14) for overbought/oversold conditions
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    rsi = np.zeros_like(close)
    rsi[:] = 50.0  # default neutral
    
    if len(gain) >= 14:
        # Initial average
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        # Wilder smoothing
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            
            # Calculate RSI
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i+1] = 100 - (100 / (1 + rs))
            else:
                rsi[i+1] = 100
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align daily pivot levels to 6-hour timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        rsi_val = rsi[i]
        
        if position == 1:  # Long
            # Exit: price returns to pivot point
            if price <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to pivot point
            if price >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Fade at R3: price at R3 with RSI > 70 (overbought) -> short
            if abs(price - r3_aligned[i]) < (r4_aligned[i] - r3_aligned[i]) * 0.02 and rsi_val > 70:
                position = -1
                signals[i] = -0.25
            # Fade at S3: price at S3 with RSI < 30 (oversold) -> long
            elif abs(price - s3_aligned[i]) < (s3_aligned[i] - s4_aligned[i]) * 0.02 and rsi_val < 30:
                position = 1
                signals[i] = 0.25
            # Breakout at R4: price breaks above R4 with volume expansion -> long
            elif price > r4_aligned[i] and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Breakout at S4: price breaks below S4 with volume expansion -> short
            elif price < s4_aligned[i] and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals