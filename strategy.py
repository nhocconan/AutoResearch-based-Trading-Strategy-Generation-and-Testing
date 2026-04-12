#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_v3
Hypothesis: Use daily Camarilla pivot levels with volume confirmation and trend filter (EMA50) to capture breakouts.
Long when price breaks above H3 with volume > 1.5x average and price above EMA50.
Short when price breaks below L3 with volume > 1.5x average and price below EMA50.
Designed for 4h timeframe to target 20-40 trades per year, minimizing fee drag while capturing strong moves.
Works in bull (follow upward breakouts) and bear (follow downward breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_v3"
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
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for volatility
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                               np.absolute(np.roll(close_1d, 1) - low_1d)))
    # Set first TR to high-low
    tr[0] = high_1d[0] - low_1d[0]
    
    # Calculate pivot points (using previous day)
    pivot = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    range_ = np.roll(high_1d, 1) - np.roll(low_1d, 1)
    
    # Camarilla levels
    H3 = pivot + (range_ * 1.1 / 2)
    L3 = pivot - (range_ * 1.1 / 2)
    H4 = pivot + (range_ * 1.1)
    L4 = pivot - (range_ * 1.1)
    
    # Align Camarilla levels to 4h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # EMA50 on 4h for trend filter
    if len(close) >= 50:
        ema50 = close.astype(np.float64)
        alpha = 2 / (50 + 1)
        for i in range(1, len(close)):
            ema50[i] = alpha * close[i] + (1 - alpha) * ema50[i-1]
    else:
        ema50 = np.full(len(close), np.nan)
    
    # Volume average (20-period)
    vol_ma = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume condition: current volume > 1.5x average
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = high[i] > H3_aligned[i]
        short_breakout = low[i] < L3_aligned[i]
        
        # Trend filter
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        # Entry logic
        long_entry = long_breakout and vol_ok and uptrend
        short_entry = short_breakout and vol_ok and downtrend
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = low[i] < L3_aligned[i] or not uptrend
        short_exit = high[i] > H3_aligned[i] or not downtrend
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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