#!/usr/bin/env python3
"""
4h_OpeningGap_Reversal_With_Volume_Confirmation
Hypothesis: Exploit overnight gaps in BTC/ETH by fading the opening gap with volume confirmation.
Works in both bull and bear markets as gaps often represent overextended moves that reverse.
Uses 1d opening gap (today's open vs yesterday's close) and fades it when accompanied by
above-average volume. Trend filter uses 1d EMA50 to avoid trading against strong trends.
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
    open_price = prices['open'].values
    
    # Get 1d data for gap calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d opening gap: (today's open - yesterday's close) / yesterday's close
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    
    gap_percent = np.zeros_like(open_1d)
    for i in range(1, len(open_1d)):
        gap_percent[i] = (open_1d[i] - close_1d[i-1]) / close_1d[i-1]
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[:] = np.nan
    if len(close_1d) >= 50:
        k = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * k + ema_50_1d[i-1] * (1 - k)
    
    # Align 1d indicators to 4h timeframe
    gap_percent_aligned = align_htf_to_ltf(prices, df_1d, gap_percent)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # Warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(gap_percent_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: gap down (negative gap) with volume spike and above EMA50
            if gap_percent_aligned[i] < -0.005 and vol_spike[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: gap up (positive gap) with volume spike and below EMA50
            elif gap_percent_aligned[i] > 0.005 and vol_spike[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: gap filled or trend change or max 8 bars hold
            if bars_since_entry >= 8 or gap_percent_aligned[i] > 0 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: gap filled or trend change or max 8 bars hold
            if bars_since_entry >= 8 or gap_percent_aligned[i] < 0 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_OpeningGap_Reversal_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0