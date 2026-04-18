#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_Volume
Hypothesis: Price breaking above Camarilla R1 or below S1 with volume confirmation and trend filter (1d EMA34) provides high-probability entries. Works in bull markets by buying R1 breaks above EMA34, and in bear markets by selling S1 breaks below EMA34. Volume > 1.5x 20-period average confirms institutional interest. Targets 20-30 trades/year via strict confluence of price level, volume, and trend alignment.
"""

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (Range * 1.1 / 12)
    # S1 = C - (Range * 1.1 / 12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (wait for bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1, with volume, and above EMA34 (uptrend)
            if (close[i] > r1_aligned[i] and vol_confirm[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1, with volume, and below EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] and vol_confirm[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below R1 or below EMA34 (trend change)
            if (close[i] < r1_aligned[i] or 
                close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or above EMA34 (trend change)
            if (close[i] > s1_aligned[i] or 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0