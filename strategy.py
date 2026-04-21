#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Volume_Filter_v1
Hypothesis: Breakout above pivot R1 (long) or below S1 (short) on 1d timeframe, confirmed by 12h volume spike and RSI momentum.
Works in both bull and bear by capturing breakouts from daily pivot levels with momentum confirmation.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Load 12h data for volume and RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.values
    
    # Align to 12h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or np.isnan(volume_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Volume filter: current 12h volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = volume_12h_aligned[i-20:i].mean()
            volume_ok = volume_12h_aligned[i] > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: breakout above R1 with bullish momentum and volume
            if (price > r1_aligned[i] and 
                rsi_12h_aligned[i] > 50 and volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below S1 with bearish momentum and volume
            elif (price < s1_aligned[i] and 
                  rsi_12h_aligned[i] < 50 and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reach R2 or reverse below R1
            if price >= r2_aligned[i] or price <= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reach S2 or reverse above S1
            if price <= s2_aligned[i] or price >= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1S1_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0