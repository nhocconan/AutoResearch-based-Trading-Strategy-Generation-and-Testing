#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_Volume_1dATRStop
Hypothesis: Breakouts beyond daily pivot R1/S1 levels with volume confirmation and 1d ATR stop loss capture momentum while limiting downside. Designed for 20-40 trades/year on 4h timeframe with low trade frequency to minimize fee drift. Works in bull/bear markets by requiring volume spike and using volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot and ATR calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points using standard formula
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Using previous 1d bar's data to avoid look-ahead
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Shift by 1 to use previous 1d bar's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # Calculate 1d ATR (14) for stop loss
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr = atr_14_1d_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike
            if price > r1_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 with volume spike
            elif price < s1_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to S1 or hits ATR stop
            if price <= s1_val or price <= entry_price - 1.5 * atr:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to R1 or hits ATR stop
            if price >= r1_val or price >= entry_price + 1.5 * atr:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Pivot_R1S1_Breakout_Volume_1dATRStop"
timeframe = "4h"
leverage = 1.0