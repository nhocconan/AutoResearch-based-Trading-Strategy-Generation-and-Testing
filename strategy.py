#!/usr/bin/env python3
"""
12h_1D_Camarilla_Pivot_R1S1_Breakout_Volume
Hypothesis: Uses 1-day Camarilla pivot levels (R1, S1) for entry with volume confirmation.
Trades on 12h timeframe with 1-day trend filter to avoid counter-trend entries.
Designed for low trade frequency (15-25/year) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot and support/resistance levels
    pivot = np.full(len(high_1d), np.nan)
    r1 = np.full(len(high_1d), np.nan)
    s1 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Use previous day's OHLC to avoid look-ahead
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        r1[i] = pivot[i] + (prev_high - prev_low) * 1.1 / 12.0
        s1[i] = pivot[i] - (prev_high - prev_low) * 1.1 / 12.0
    
    # 1-day EMA34 for trend filter
    ema34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34[i] = close_1d[i] * alpha + ema34[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 1-day data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need volume MA and EMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and 1-day uptrend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and 1-day downtrend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below S1 or 1-day trend turns down
            if (close[i] < s1_aligned[i] or close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 or 1-day trend turns up
            if (close[i] > r1_aligned[i] or close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0